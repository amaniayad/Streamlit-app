import streamlit as st
import reveal_slides as rs

sample_markdown = r"""
## Machine Learning Operations (MLOps)
`Presented By:`<br> 
`Belmiloud Maroua`<br>
`Ayad Amani`<br> 
`Senkadi Khawla`
---
## Project Timeline:

- Definition of MLOPs
- Objectives
- Steps of Mlops
- MLOps platforms
- Conclusion
---
## Definition:
`Machine Learning Operations (MLOps)` is a set of practices that combines Machine Learning (ML) and DevOps (Development and Operations) to automate the end-to-end process of deploying and managing ML models in production.

---
## Objectives:
1. Increase the reliability and efficiency of ML deployments.
2. Reduce the time it takes to get ML models into production.
3. Improve the collaboration between ML engineers, data scientists, and DevOps engineers.

---
## Steps of Mlops
`Data preparation:` This involves cleaning and formatting the data that will be used to train the ML model.<br>
`Model training:` This involves using an algorithm to learn from the data and create a model that can make predictions.<br>
`Model evaluation:` This involves testing the model to see how accurate it is.<br>
`Model deployment:` This involves making the model available to users so that they can use it to make predictions.<br>
`Model monitoring:` This involves tracking the performance of the model over time and making adjustments as needed.
---
## MLOps platforms:
There are commercial MLOps platforms such as:
`1.Amazon SageMaker:` Amazon SageMaker is an MLOps platform that is offered by Amazon Web Services (AWS)<br>
`2.Azure Machine Learning:` Azure Machine Learning is an MLOps platform that is offered by Microsoft Azure.<br>
`3.TGoogle Cloud ML Engine:` Google Cloud ML Engine is an MLOps platform that is offered by Google Cloud Platform (GCP).<br>
Theye all provide a set of tools for building, deploying, and managing ML models in production.
--
In addition to these tools, there are also a number of open source MLOps frameworks available, Some of them include:<br>
`1.Kubeflow`<br>
`2.MLflow`<br>
`3.TensorFlow Extended (TFX)`
---
## Kubeflow:
Kubeflow is an open-source platform built on Kubernetes, designed to simplify the deployment, scaling, and management of machine learning workflows in production.
--
`Features:`

|  |         |
|----:|-----------------|
| 1   |Pipeline Orchestration          |
| 2   |Model Serving    |
| 3   |Experiment Tracking      |
| 4   |Hyperparameter Tuning        |

`Use Cases:` Kubeflow is suitable for organizations looking to deploy and manage machine learning workflows at scale using Kubernetes.
---
## MLflow:
MLflow is an open-source platform for managing the end-to-end machine learning lifecycle, including experimentation, reproducibility, deployment, and monitoring.
--
`Features:`

|  |         |
|----:|-----------------|
| 1   |Experiment Tracking          |
| 2   |Model Packaging    |
| 3   |Model Deployment      |
| 4   |Model Registry        |

`Use Cases:` MLflow is suitable for organizations looking for a unified platform to manage their machine learning workflows, from experimentation to production deployment.
---
## TensorFlow Extended (TFX):
TensorFlow Extended (TFX) is an end-to-end platform for deploying production-ready machine learning pipelines powered by TensorFlow.
--
`Features:`

|  |         |
|----:|-----------------|
| 1   |Pipeline Construction          |
| 2   |Model Versioning    |
| 3   |Model Evaluation      |
| 4   |Integration with TensorFlow      |

`Use Cases:` TFX is suitable for organizations leveraging TensorFlow for building and deploying machine learning models in production environments.
---
## Conclusion:
The open source MLOps frameworks provide a good starting point for building an MLOps platform. However, they may not be suitable for all organizations. If you need a more comprehensive MLOps platform, you may need to consider a commercial MLOps platform.
---
# Thanks for your attention !
---

"""
st.markdown("## MLOPS presentation")
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
    vslidePos = st.number_input("Vertical Slide Position", value=0)
    fragPos = st.number_input("Fragment Position", value=-1)
    overview = st.checkbox("Show Overview", value=False)
    paused = st.checkbox("Pause", value=False)

                    
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
                                    "indexv": vslidePos, 
                                    "indexf": fragPos, 
                                    "paused": paused, 
                                    "overview": overview 
                                    }, 
                    markdown_props={"data-separator-vertical":"^--$"}, 
                    key="foo")
