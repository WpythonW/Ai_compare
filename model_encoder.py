from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer
import pickle
from sentence_transformers import InputExample, losses
from torch.utils.data import DataLoader
import numpy as np


model = SentenceTransformer("model/")

def find_sim(distances, Lub_sorted_brands_text, Lub_sorted_brands_mass, Lub_sorted_brands_unit, Lub_sorted_brands_count, Lub_sorted_brands_ind, Matr_sorted_brands_text, Matr_sorted_brands_mass, Matr_sorted_brands_unit, Matr_sorted_brands_count, Matr_sorted_brands_ind, threshold):
  result = []
  count_lubb = 0
  count_match = 0
  exc = []
  for i in range(distances.shape[0]):
      current_Lub_name_compared_to_all_Matrs = distances[i]

      sorted_index = np.argsort(current_Lub_name_compared_to_all_Matrs)
      checked = False

      found = False
      mass = False
      unit = False
      count = False

      for ind in sorted_index[:]:
        if Lub_sorted_brands_mass[i] and Matr_sorted_brands_mass[ind] and Lub_sorted_brands_mass[i] != Matr_sorted_brands_mass[ind]:
          continue
        elif Lub_sorted_brands_mass[i] and Matr_sorted_brands_mass[ind] and Lub_sorted_brands_mass[i] == Matr_sorted_brands_mass[ind]:
          mass = True
        if Lub_sorted_brands_unit[i] and Matr_sorted_brands_unit[ind] and Lub_sorted_brands_unit[i] != Matr_sorted_brands_unit[ind]:
          continue
        elif Lub_sorted_brands_unit[i] and Matr_sorted_brands_unit[ind] and Lub_sorted_brands_unit[i] == Matr_sorted_brands_unit[ind]:
          unit = True
        if Lub_sorted_brands_count[i] and Matr_sorted_brands_count[ind] and Lub_sorted_brands_count[i] != Matr_sorted_brands_count[ind]:
          continue
        elif Lub_sorted_brands_count[i] and Matr_sorted_brands_count[ind] and Lub_sorted_brands_count[i] == Matr_sorted_brands_count[ind]:
          count = True

        if mass and unit and count and (ind not in exc):
          found = True
          sorted_min = current_Lub_name_compared_to_all_Matrs[ind]
          fn_index = ind
          break

      if found and sorted_min < threshold:
        checked = True

      if checked:
        result.append((Lub_sorted_brands_ind[i], Matr_sorted_brands_ind[fn_index]))
        exc.append(fn_index)
        count_match += 1


      else:
        result.append((Lub_sorted_brands_ind[i], None))
      count_lubb += 1
  return (result, count_match/count_lubb)

with open('Lub_sorted_brands.pickle', 'rb') as f:
    Lub_sorted_brands = pickle.load(f)

with open('Matr_sorted_brands.pickle', 'rb') as f:
    Matr_sorted_brands = pickle.load(f)

availiable_brands = Lub_sorted_brands.keys()
len(availiable_brands)

final_res = []
for i in availiable_brands:
    if Lub_sorted_brands[i] and Matr_sorted_brands[i]:
        Lub_sorted_brands_text = [x[0] for x in Lub_sorted_brands[i]]
        Matr_sorted_brands_text = [x[0] for x in Matr_sorted_brands[i]]

        print(len(Lub_sorted_brands_text))

        Lub_sorted_brands_mass = [x[1] for x in Lub_sorted_brands[i]]
        Matr_sorted_brands_mass = [x[1] for x in Matr_sorted_brands[i]]

        Lub_sorted_brands_unit = [x[2] for x in Lub_sorted_brands[i]]
        Matr_sorted_brands_unit = [x[2] for x in Matr_sorted_brands[i]]

        Lub_sorted_brands_count = [x[3] for x in Lub_sorted_brands[i]]
        Matr_sorted_brands_count = [x[3] for x in Matr_sorted_brands[i]]

        Lub_sorted_brands_ind = [x[5] for x in Lub_sorted_brands[i]]
        Matr_sorted_brands_ind = [x[5] for x in Matr_sorted_brands[i]]

        #print(Lub_sorted_brands_unit)
        #print(indices_g)

        #break
        #rint(Lub_sorted_brands_ind)
        #print(Matr_sorted_brands_ind)

        embeddings1 = model.encode(Lub_sorted_brands_text)
        embeddings2 = model.encode(Matr_sorted_brands_text)
        
        #try:
        distances = cdist(embeddings1, embeddings2, 'cosine')
        
        threshold = 0.15
        res, ratio = find_sim(distances, Lub_sorted_brands_text, Lub_sorted_brands_mass, Lub_sorted_brands_unit, Lub_sorted_brands_count, Lub_sorted_brands_ind, Matr_sorted_brands_text, Matr_sorted_brands_mass, Matr_sorted_brands_unit, Matr_sorted_brands_count, Matr_sorted_brands_ind, threshold)
        #except Exception as e:
        #print(e)
        #break
        #res = [(x, None) for x in Lub_sorted_brands_ind]
        print(ratio, i)

        final_res += res
    else:
        Lub_sorted_brands_ind = [x[5] for x in Lub_sorted_brands[i]]
        for i in Lub_sorted_brands_ind:
           final_res.append((i, None))

with open('final_res.pickle', 'wb') as f:
    pickle.dump(final_res, f)