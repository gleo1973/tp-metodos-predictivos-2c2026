# 📊 Trabajo Práctico Grupal · Métodos Predictivos para la Gestión

> **Tecnicatura Universitaria en Gestión y Análisis de Datos**  
> _Facultad de Ciencias Económicas · Universidad de Buenos Aires (FCE - UBA)_  
> **2.er Cuatrimestre 2026**

---

## 👥 Integrantes

| Nombre completo        | Registro |  Usuario GitHub   |
| :--------------------- | :------: | :---------------: |
| **Martina Bisio**      |  921302  |  [Tinirainboom]   |
| **Katherine Freites**  |  921314  | [Facuecoyese-bot] |
| **Didier Detchemendy** |  921341  |  [didetchemendy]  |
| **Santiago Mesones**   |  921306  |  [Santiago1614]   |
| **Leonardo Gomez**     |  921318  |    [gleo1973]     |
| **Lorenzo Cairola**    |  921307  | [lorenzocairola]  |

---

## 📝 Descripción del Proyecto

El presente trabajo práctico tiene como objetivo el análisis, modelado y predicción de la popularidad de artículos de noticias web utilizando el dataset **Online News Popularity** del repositorio de Machine Learning de la UCI.

A partir de un conjunto heterogéneo de características (métricas de contenido, metadatos temporales, presencia multimedia, análisis de sentimiento y asignación temática mediante LDA), se aborda un flujo metodológico riguroso que abarca desde la depuración estadística de los datos hasta la comparación del poder predictivo entre enfoques tradicionales y algoritmos avanzados de Machine Learning.

### 🛠️ Tecnologías y Librerías Utilizadas

- **Lenguaje:** Python 3.13 (para adelante)
- **Auditoría y Estadística:** `statsmodels`, `scipy.stats`
- **Machine Learning:** `scikit-learn` (Random Forest, Gradient Boosting)
- **Manipulación de Datos:** `pandas`, `numpy`
- **Visualización:** `matplotlib`, `seaborn`

---

## 📂 Estructura del Repositorio

```text
tp-laboratorio-mcag-1c2026/
│
├── Metodos_Predictivos_TP_2026.ipynb   # Notebook principal con el código y análisis
├── README.md                           # Documentación general del repositorio
│
├── datos/                              # Almacenamiento local del dataset (si aplica)
│   └── OnlineNewsPopularity.csv
│
└── informe/                            # Documentación formal y entregables de texto
    ├── figuras/                        # Gráficos e imágenes exportadas (ej. VIF, Importancia)
    └── Informe_Final_Predictivos.md   # Reporte escrito final del Trabajo Práctico
```

---

## 🚀 Instrucciones de Ejecución

Para reproducir el análisis estadístico y las corridas de los modelos predictivos de forma local:

1. **Clonar el repositorio:**

   ```bash
   git clone https://github.com/tp-laboratorio-mcag-1c2026.git
   cd tp-laboratorio-mcag-1c2026
   ```

2. **Instalar las dependencias requeridas:**
   Asegurate de contar con los paquetes necesarios ejecutando la celda inicial del archivo `.ipynb` o mediante la terminal:

   ```bash
   pip install ucimlrepo pandas numpy statsmodels scipy scikit-learn matplotlib seaborn
   ```

3. **Ejecutar el entorno:**
   Abrir el entorno de Jupyter Notebook o Google Colab y ejecutar las celdas de forma secuencial desde la sección **1 (Carga y Exploración Inicial)** hasta las **Conclusiones**.
