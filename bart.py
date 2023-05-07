from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('bert-base-multilingual-cased', device='cuda')
from sentence_transformers import InputExample, losses
from torch.utils.data import DataLoader

model.save("./model/")
# Создание списка примеров для обучения
# Каждый пример содержит пару предложений и метку схожести
train_examples = [InputExample(texts=['корм для котёнок порода британский короткошёрстный', 
                                      'kitten british shorthair сухой корм для котёнок порода британский гладкошёрстный'], 
                                      label=1.0),
                  InputExample(texts=['влажный корм для взрослый кошка весь порода говядина соус уп', 
                                      'adult cat влажный корм для взрослый кошка говядина соус'], 
                                      label=1.0),
                  InputExample(texts=['мясной ломтик влажный корм для взрослый кошка весь порода утка уп пауч', 
                                      'мясной ломтик влажный корм для взрослый кошка утка соус пауч'], 
                                      label=1.0),
                  InputExample(texts=['аппетитный кусочек влажный корм для взрослый кошка весь порода говядина желе уп', 
                                      'аппетитный кусочек влажный корм для взрослый кошка индейка желе пауч'], 
                                      label=0.0),
                  InputExample(texts=['sensat влажный корм для взрослый кошка весь порода курица желе морковь уп', 
                                      'sensat влажный корм для взрослый кошка курица морковь желе пауч'], 
                                      label=1.0),
                  InputExample(texts=['аппетитный кусочек влажный корм для взрослый кошка весь порода кролик желе', 
                                      'аппетитный кусочек влажный корм для взрослый кошка курица томат желе пауч'], 
                                      label=0.0),
                  InputExample(texts=['корм для взрослый собака средний крупный порода ягнёнок', 
                                      'сухой корм для взрослый собака средний крупный порода говядина рис'], 
                                      label=0.0),
                  InputExample(texts=['аппетитный кусочек влажный корм для взрослый кошка весь порода говядина желе уп', 
                                      'аппетитный кусочек влажный корм для взрослый кошка курица томат желе пауч'], 
                                      label=0.0),
                  
                  ]

#корм для взрослых стерилизованных кошек всех пород, лосось пшеница 1,5 кг
#сухой корм для взрослых стерилизованных кошек с лососем и пшеницей - 1,5 кг

# Определение функции потерь
train_loss = losses.CosineSimilarityLoss(model)

# Создание загрузчика данных для обучения
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)

# Дообучение модели на новых данных
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=100)

model.save("./model/")