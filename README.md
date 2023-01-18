# Tabular: Deep learning models vs boostings
### По мотивам статьи [Why do tree-based models still outperform deep learning on typical tabular data?](https://hal.science/hal-03723551/document)

В этом репозитории я реализую и запускаю четыре вида моделей (Catboost, MLP, TabNet и ResNet) на нескольких задачах регрессии, предложенных авторами статьи. В качестве скоринга во всех датасетах используется `r2_score`. В качестве сравнения я использую неинформативные признаки (для наборов данных с `[1, 5, 20]_trash_features` в названии) и случайный поворот матрицы признаков (`_rotated`)

В отличие от авторов, использующих случайный поиск, я использую отбор гиперпараметров с помощью `Optuna`, что должно уменьшить число запусков до сходимости -- с 200 у авторов до 100 у меня. Логирование осуществляется в wandb, а также в папку `images_final`.

## Для запуска:
1. Установите зависимости: `pip3 install -r requirements.txt`
2. Выберите нужный конфиг (датасет, модель, число итераций отбора параметров) в теле файла `src/test.py`
3. Запустите обучение: `python3 src/test.py`
4. При необходимости, сгенерируйте новые данные с помощью файлов `src/make_rotation.py`, `src/make_trash_features.py`, добвьте новые модели (реализуйте новые функции `objective_XXX(trial)` в `src/test.py`)

## Результаты
### Сравннение моделей

#### Обычный датасет, без поворотов, без случайных признаков
wine | fifa
:-:|:-:
![wine_quality](plots/00_wine_quality.png "wine_quality") | ![fifa](plots/01_fifa.png "fifa")

#### Датасет со случайным поворотом
wine | fifa
:-:|:-:
![wine_quality](plots/02_wine_quality,_rotated.png "wine_quality") | ![fifa](plots/03_fifa,_rotated.png "fifa")

#### Датасет с одним случайным признаком
wine | fifa
:-:|:-:
![wine_quality](plots/04_wine_quality,_1_random_feature.png "wine_quality") | ![fifa](plots/05_fifa,_1_random_feature.png "fifa")

#### Датасет с пятью случайными признаками
wine | fifa
:-:|:-:
![wine_quality](plots/06_wine_quality,_5_random_features.png "wine_quality") | ![fifa](plots/07_fifa,_5_random_features.png "fifa")

#### Датасет с двадцатью случайными признаками
wine | fifa
:-:|:-:
![wine_quality](plots/08_wine_quality,_20_random_features.png "wine_quality") | ![fifa](plots/09_fifa,_20_random_features.png "fifa")

#### Датасет с двадцатью случайными признаками и поворотом
wine | fifa
:-:|:-:
![wine_quality](plots/10_wine_quality,_20_random_features,_rotated.png "wine_quality") | ![fifa](plots/11_fifa,_20_random_features,_rotated.png "fifa")

### Примеры запусков
    Boosting | MLP 
:-:|:-:
![boosting](images_final/fifa_1_trash_boosting_0.6772.png "boosting") | ![mlp](images_final/fifa_1_trash_mlp_0.6517.png "mlp") 
ResNet | TabNet
![resnet](images_final/fifa_1_trash_resnet_0.6551.png "resnet") | ![tabnet](images_final/fifa_1_trash_tabnet_0.6632.png "tabnet") |