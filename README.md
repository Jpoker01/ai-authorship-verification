# Authorship verification using selected artificial intelligence methods

This project folder contains all work performed within the scope of the diploma thesis 'Authorship verification using selected artificial intelligence methods'.  This application allows users to verify whether a given text was written by a specific author using a machine learning classifier.

The final deployed website can be accessed through: www.verifyauthor.dev  
The diploma thesis (in Czech) is available at: <LINK_TO_BE_ADDED>  

## Quickstart guide

### Running frontend application

Run from the root of the frontend folder:  
```bash
npm install
npm run build
npm run preview
```

Navigate to `http://localhost:4173` (or the URL shown in your terminal)  

### Running backend application

Run from the backend folder:  
```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip setuptools wheel 
pip install -r "requirements.txt"
uvicorn main:app --reload   
```


## Project structure

The work can be broken into two main parts:
* **Experiments** - All of the Jupyter notebooks where different methods of AI are utilized and experimented with to find the optimal solution
* **Web application** - The web application where a chosen authorship verification model is deployed and where the user can verify the authorship of two texts.

### Experiments (`/experiments`)
Lists all the experiments done for this diploma thesis.

 * **requirements.txt** - The requirements for reproducing the implementation environment
 * **/conf** - Space for configurations
     * **/base** - Shared configuration like parameters
     * **/local** - Local configurations such as credentials
 * **/notebooks** - Jupyter notebooks - naming convention "YYYYMMDD_developerinitials_description"
   * **/dataset** - Notebooks related to dataset processing and analysis
   * **/graph** - Notebooks related to experiments utilizing integrated syntactic graphs
   * **/llm** - Notebooks utilizing LLMs
   * **/traditional** - Notebooks utilizing traditional BOW/TF-IDF representations for experiments 
   * **/transformer** - Notebooks utilizing transformer-based models for experiments
 * **/results** - Final analysis documents
   * **/llm** - Lists all the LLM experiment results (done before utilizing MLFlow)
   * **/prilohaC_MLFlow_all_experiments_export.xlsx** - Lists all the MLFlow results for other experiments than those utilizing LLMs
     
###  Web application (`/webapp`)
Contains the source code for the frontend and backend of the final web application solution

* **/frontend** - Contains the source code for the frontend of the web application
* **/backend** - Contains the source code for the backend of the web application
