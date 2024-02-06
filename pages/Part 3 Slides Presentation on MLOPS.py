import streamlit as st
import reveal_slides as rs

sample_markdown = r"""
# AERO DELAY 
`Presented By:`<br> 
`Belmiloud Maroua`<br>
`Ayad Amani`<br> 
`Senkadi Khawla`
---
# Project Timeline

- Background And Objectifs Of the Project
- Data Understanding
- Modelling
- Evaluation
- Deployment

---
## Background And Objectifs

Using machine learning, this project helps predict and address flight delays, making travel smoother for passengers and the aviation industry.
---
## Data Understanding
| `Source Of The Data`   | Airline On-Time Performance and Causes of Flight Delays dataset reported by major U.S. airlines |
| ------------------------------------- | ---------------------------------------------------------- |
| `Shape` | Labeled dataset :(100,000,9)<br>Unlabeled dataset :(100,000,8)                             |
| `Label Information`    | The class is represented by the "dep_delayed_15min" column, is of binary type, indicating whether a flight is delayed by 15 minutes or more (yes or no).                              |

---
## Features 
|  |         |
|----:|-----------------|
| 1   | Month           |
| 2   | DayOfMonth      |
| 3   | DayOfWeek       |
| 4   | DepTime         |
| 5   | UniqueCarrier   |
| 6   | Origin          |
| 7   | Dest            |
| 8   | Distance        |

---
## Preprocessing
| `Handling Missing Values`   | No missing data was detected. |
| ------------------------------------- | ---------------------------------------------------------- |
| `Handling Outliers` | Employing the Interquartile Range (IQR) method to handle outliers in the feature “Distance”                             |
| `NUMERISATION`    | Categorical features were encoded using the Ordinal Encoder from the scikit-learn library<br>'c-' prefixes in the features were removed<br>the data was converted to float format for consistency.|
---
## Preprocessing
| `Feature Scaling`   | The Min-Max Scaler from scikit-learn was applied to normalize the values within the range of [0, 1]. |
| ------------------------------------- | ---------------------------------------------------------- |
| `Feature Selection` | None of the features exhibit high correlation with the class, So all the available features are used for model training.                             |
| `Data Splitting`    | Employing a standard train-test split methodology.|
---
## Modeling 
## Semi-Supervised Learning
Semi-supervised learning is a machine learning approach that sits between supervised and unsupervised learning.   
 Unlike supervised learning, which relies entirely on labeled data, and unsupervised learning, which deals with unlabeled data
---
## Approaches
### 1/ Self-Training
The self-training algorithm starts with a model trained on a small set of labeled data. It then uses this model to make predictions on unlabeled data. 
### 2/ Label Propagation
Label propagation involves creating a graph representation of the dataset, where nodes are instances and edges connect similar instances. Labels from labeled instances are then propagated to their graph neighbors.
---
## Self Training:
<img src="./self.png" alt="self training" width="1500" height="800">
---
## Label Propagation:
<img src="./label.png" alt="label propagation" width="1500" height="800">
---
## Evaluation Metrics:
| | |
| ------------------------------------- | ---------------------------------------------------------- |
| `accuracy`   | `recall` |
| `precision` | `F1-score`    |

We focus on precision as a priority metric, aiming to minimize false negatives. Predicting a flight as not delayed when it is carries less impact than risking passengers missing their flight due to an incorrect delay prediction.
---
## Model Results:

| `1)K-Nearest Neighbors (KNN)`| `2)Naive Bayes`              |
|--------------------------------|--------------------------------|
| Accuracy : 59%          | Accuracy : 80%           |
| Precision : 74%           | Precision : 65%           |
| Recall : 59%         | Recall : 80%           |
| F1-Score : 63%           | F1-Score : 72%           |

---

| `3)Logistic Regression`| `4)Decision Tree`              |
|--------------------------------|--------------------------------|
| Accuracy : 80%          | Accuracy : 52%           |
| Precision : 73%           | Precision : 73%           |
| Recall : 80%         | Recall : 52%           |
| F1-Score : 72%           | F1-Score : 57%           |
---
| `5)SVM`| `6)Label propagation`              |
|--------------------------------|--------------------------------|
| Accuracy : 80%          | Accuracy : 78%           |
| Precision : 75%           | Precision : 72%           |
| Recall : 80%         | Recall : 78%           |
| F1-Score : 72%           | F1-Score : 74%           |
---
## The best Model:
<img src="./result.png" alt="label propagation" width="900" height="600">
The SVM model excels with higher precision, making it the preferred choice over 
the self-training model.
---
## Conclusion:
In conclusion, the model's accuracy may be influenced by external factors such as unpredictable weather or sudden airport changes. Additionally, it's important to note that our study is limited to the available dataset and its features.
---
# Thanks for your attention !
---
"""
st.markdown("## Aero delay project presentation")
with st.sidebar:
    st.header("Component Parameters")
    theme = st.selectbox("Theme", ["black", "black-contrast", "blood", "dracula", "moon", "white", "white-contrast", "league", "beige", "sky", "night", "serif", "simple", "solarized"])
    height = st.number_input("Height", value=500)
    st.subheader("Slide Configuration")
    content_height = st.number_input("Content Height", value=900)
    content_width = st.number_input("Content Width", value=900)
    scale_range = st.slider("Scale Range", min_value=0.0, max_value=5.0, value=[0.1, 3.0], step=0.1)
    margin = st.slider("Margin", min_value=0.0, max_value=0.8, value=0.1, step=0.05)
    plugins = st.multiselect("Plugins", ["highlight", "katex", "mathjax2", "mathjax3", "notes", "search", "zoom"], [])
    st.subheader("Initial State")
    hslidePos = st.number_input("Horizontal Slide Position", value=0)
    overview = st.checkbox("Show Overview", value=False)
    paused = st.checkbox("Pause", value=False)

# Add the streamlit-reveal-slide component to the Streamlit app.                    
currState = rs.slides(sample_markdown, 
                    height=height, 
                    theme=theme, 
                    config={
                            "width": content_width, 
                            "height": content_height, 
                            "minScale": scale_range[0], 
                            "center": True, 
                            "maxScale": scale_range[1], 
                            "margin": margin, 
                            "plugins": plugins
                            }, 
                    initial_state={
                                    "indexh": hslidePos, 
                                    "paused": paused, 
                                    "overview": overview 
                                    },  
                    key="foo")
