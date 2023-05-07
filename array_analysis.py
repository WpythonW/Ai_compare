import pickle 
import pandas as pd
import csv
import locale
def getpreferredencoding(do_setlocale = True):
    return "UTF-8"
locale.getpreferredencoding = getpreferredencoding


# Определяем функцию для преобразования строки
def process_string(s):
    sp = s.split()
    return " ".join(sp).lower()

with open('final_res.pickle', 'rb') as f:
    final_res = pickle.load(f)
#print(final_res)
print(len(final_res))

def find_dict(i, D):
   for k in D.keys():
      for sent in D[k]:
        if int(sent[-1]) == i:
           return sent[0].replace("паучахнуть", "пауч").replace("паучий", "пауч")


Matr = []
df_tm = pd.read_csv('output.csv', header=None, encoding="utf8", delimiter=',')

# Применяем функцию к столбцу с индексом 0
df_tm.iloc[:, 0] = df_tm.iloc[:, 0].apply(process_string)

Matr = df_tm.iloc[:, 0].copy()
prices_Matr = df_tm.iloc[:, 1].copy()


Lub = []
df_t = pd.read_csv('products.csv', encoding="utf8", delimiter=',')

# Применяем функцию к столбцу 'column_name'
df_t['Наименование товара'] = df_t['Наименование товара'].apply(process_string)

Lub = df_t["Наименование товара"].copy()
prices_Lub = df_t["Текущая цена с учетом скидки, руб."].copy()

#print(len(Lub), len(prices_Lub))
#print(len(Matr), len(prices_Matr))

df_Lub_sorted_brands = pd.read_csv('sorted_brands_Lub.csv', encoding="utf8", delimiter=',')
Lub_sorted_brands = df_Lub_sorted_brands.to_dict('list')

df_Matr_sorted_brands = pd.read_csv('sorted_brands_Matr.csv', encoding="utf8", delimiter=',')
Matr_sorted_brands = df_Matr_sorted_brands.to_dict('list')

for arr in Lub_sorted_brands.keys():
  arr_new = []
  for nm in Lub_sorted_brands[arr]:
    if not type(nm) == float:
      #print(nm)
      nm = nm.split('@')
      sent = nm[0].split()
      arr_new.append((" ".join(sent[:-4]), sent[-4], sent[-3], sent[-2], sent[-1], nm[1]))      
      #print(arr_new[-1])

    else:
      break
  Lub_sorted_brands[arr] = arr_new

for arr in Matr_sorted_brands.keys():
  arr_new = []
  for nm in Matr_sorted_brands[arr]:
    if not type(nm) == float:
      #print(nm)
      nm = nm.split('@')
      sent = nm[0].split()
      arr_new.append((" ".join(sent[:-4]), sent[-4], sent[-3], sent[-2], sent[-1], nm[1]))
      #print(arr_new[-1])
    else:
      break
  Matr_sorted_brands[arr] = arr_new


c = 0
fieldnames = ["Lub", "Matr", "Sim"]

try:   
    for i, j in final_res:
        if j:
            i = int(i)
            j = int(j)
            print(Lub[i])
            print(Matr[j])
            sml = input()

            if sml == "pass":
               continue
            f = open('train.csv', 'a', newline='', encoding='utf-8')
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            new_row = {'Lub': find_dict(i, Lub_sorted_brands), 'Matr': find_dict(j, Matr_sorted_brands), 'Sim': float(sml)}
            writer.writerow(new_row)

            f.close()
        c += 1

    print(c)
except:
   f.close()