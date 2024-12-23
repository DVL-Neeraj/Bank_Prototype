import numpy as np
import pandas as pd


from ..helpers import (calculate_age_group, categorical_probability,
                       cibil_score_probability,co_applicant_cibil_score_probability,
                       education_loan_opportunity, calculate_loan_size_probability,calculate_loan_duration_probability)

categories = {
    "age_group": {
        "18-24": 0.1,
        "24-30": 0.1,
        "30-35": 0.05,
        "35-40": 0.04,
        "40-45": 0.02,
        "45-50": 0.0,
        "50-55": 0.0,
        "55-60": -0.1,
    },

    "gender": {"M": 0, "F": 0},
    "education_level": {
        "High School Diploma": 0.05,
        #"None": 0.0,
        "Bachelors": 0.05,
        "Masters": 0.03,
    },
    "region": {"Urban": 0.05, "Rural": 0.0},
    "marital_status": {"Married": 0.0, "Unmarried": 0.0},
    "rm_skillset": {
        "Knowledgeable Regarding Education Loans" : 0.05,
        "Low Effort": -0.05,
        "Follows Through": 0.03,
    },
    "loan_type": {"Domestic":0.0,"International":0.0},
    "university" : {"Grade A":0.1,"Grade B":0.05,"Grade C":-1.0},
    "co_applicant":{"Available":0.0,"Not-Available":0.0},
    "co_applicant_age_group": {
        "18-24": 0.02,
        "24-30": 0.03,
        "30-35": 0.04,
        "35-40": 0.05,
        "40-45": 0.07,
        "45-50": 0.07,
        "50-55": 0.05,
        "55-60": 0.0,
        "60-70": -0.04,
    },
    #"employer": {"Unemployed": 0.0, "Employed": 0.25, "Self-Employed": 0.15},
    # "co_applicant_loan_history": {
    #     "None": 0,
    #     "Loans Repaid On Time": 0.06,
    #     "Loans Not Repaid On Time": -0.06,
    # },
    #"event": {"None": 0, "Navratri": 0.5, "Shradhsa": -0.5},    
    #"terminates_products_early": {True: -0.6, False: 0},
    #"unfavourable_employer": {True: -0.1, False: 0},
    "unfavourable_profession": {True: -0.1, False: 0},
    "kyc_status": {"Non-Compliant": -1.0, "Compliant": 0}}


def assign_label_education_loan(base_df):

    df = base_df[['age','gender', 'education_level', 'region',
       'marital_status','cibil','net_worth','kyc_status',"date"]]


    df["probability"] = 0.0
    #df["conversion_duration"] = np.random.randint(0, 50, df.shape[0])
    #df["interest_rate"] = np.round(np.random.uniform(0.01, 0.15, df.shape[0]), 2)
    
    ##columns and probabilities:
    #age
    print("calculating age group...")
    df["age_group"] = calculate_age_group(df["age"])   
    

    ###co_applicant
    print("adding co_applicant...")
    df["co_applicant"] = np.random.choice(['Available','Not-Available'],df.shape[0],p=[0.8,0.2])
    df.loc[(df["cibil"] <= 450) & (df["co_applicant"] != "Available"),"co_applicant"] = "Available"
    df.loc[(df["net_worth"] < 300000) & (df["co_applicant"] != "Available"),"co_applicant"] = "Available"

    ##applicant_networth:
    print("calculating applicant networth probability...")
    df["applicant_networth_probability"] = 0.0
    df["applicant_networth_probability"] = df["net_worth"].apply(lambda x:(x//500000)*0.01)
    df.loc[df["applicant_networth_probability"]>0.2,"applicant_networth_probability"] = 0.2
    df.loc[(df["co_applicant"]=="Available") & (df["applicant_networth_probability"] > 0.03) ,"applicant_networth_probability"] = 0.03
    df["probability"] += df["applicant_networth_probability"]

    ##cibil
    print("calculating applicant cibil probability...")  
    df["cibil_score_probability"] = 0.0
    df["cibil_score_probability"] = cibil_score_probability(df[["cibil_score_probability", "cibil","co_applicant"]])
    df.loc[(df["co_applicant"]=="Available") & (df["cibil_score_probability"] > 0.03) ,"cibil_score_probability"] = 0.03
    df["probability"] += df["cibil_score_probability"]

    ##Rm_Relation:  
    print("calculating rm relation probability...")
    df["rm_relation"] = np.random.randint(0,3,size=df.shape[0])
    df["rm_relation_probability"] = df["rm_relation"]*0.025
    df["probability"] +=  df["rm_relation_probability"] 

    ##co-applicant age
    print("calculating co applicant age...")
    df["co_applicant_age"] = df["age"].apply(lambda x: x+np.random.randint(3,31))
    df.loc[df["co_applicant"]=="Not-Available","co_applicant_age"] = 0
    df["co_applicant_age_group"] = calculate_age_group(df["co_applicant_age"])

    #df.loc[df["cibil"] <= 350,"cibil"] += 400
    #rm skillset
    print("adding other categorical features...")
    df["rm_skillset"] = np.random.choice(["Knowledgeable Regarding Education Loans",
                                            "Low Effort","Follows Through"], df.shape[0],p=[0.7,0.1,0.2])

    df["loan_type"] = np.random.choice(['Domestic','International'],df.shape[0],p=[0.6,0.4])

    df["university"] = np.random.choice(['Grade A','Grade B','Grade C'],df.shape[0],p=[0.7,0.2,0.1])

    df["co_applicant_loan_history"] = np.random.choice(['No Loan History','Loans Repaid On Time','Loans Not Repaid On Time'],df.shape[0],p=[0.15,0.7,0.15])

    #df["unfavourable_employer"] = np.random.choice([True,False],df.shape[0],p=[0.15,0.85])

    df["unfavourable_profession"] = np.random.choice([True,False],df.shape[0],p=[0.15,0.85])


    #co applicant cibil
    print("calculating co-applicant cibil probability...")
    df["co_applicant_cibil"] = np.random.randint(600,850,size=df.shape[0])
    df.loc[df["co_applicant_loan_history"] == "Loans Repaid On Time","co_applicant_cibil"] = np.random.randint(650,900,size=df.loc[df["co_applicant_loan_history"] == "Loans Repaid On Time","co_applicant_cibil"].shape[0])
    df.loc[df["co_applicant_loan_history"] == "Loans Not Repaid On Time","co_applicant_cibil"] = np.random.randint(300,600,size=df.loc[df["co_applicant_loan_history"] == "Loans Not Repaid On Time","co_applicant_cibil"].shape[0])
    df.loc[(df["co_applicant"] == "Not-Available"),"co_applicant_cibil"] = 0
    df["co_applicant_cibil_score_probability"] = 0
    df["co_applicant_cibil_score_probability"] = co_applicant_cibil_score_probability(df[["co_applicant_cibil_score_probability", "co_applicant_cibil","co_applicant"]])
    df["probability"] += df["co_applicant_cibil_score_probability"]

    ##co_applicant_networth:
    print("calculating co-applicant networth probability...")
    df["co_applicant_networth"] = np.random.randint(10,201,size=df.shape[0])*50000
    df.loc[(df["co_applicant"] == "Not-Available"),"co_applicant_networth"] = 0
    df["co_applicant_networth_probability"] = 0.0
    df["co_applicant_networth_probability"] = df["co_applicant_networth"].apply(lambda x:(x//500000)*0.01)
    df.loc[df["co_applicant_networth_probability"]>0.075,"co_applicant_networth_probability"] = 0.075
    df["probability"] += df["co_applicant_networth_probability"]


    #co_applicant_salary:
    print("calculating co-applicant salary probability...")
    df["co_applicant_salary"] = np.random.randint(6,201,size=df.shape[0])*50000
    df.loc[(df["co_applicant"] == "Not-Available"),"co_applicant_salary"] = 0
    df["co_applicant_salary_probability"] = 0.0
    df["co_applicant_salary_probability"] = df["co_applicant_salary"].apply(lambda x:(x//500000)*0.01)
    df.loc[df["co_applicant_salary_probability"]>0.07,"co_applicant_salary_probability"] = 0.07
    df["probability"] += df["co_applicant_salary_probability"]

    ##co_applicant_existing_emi:
    df["co_applicant_existing_emi"] = np.random.randint(0,25,size=df.shape[0])*1000
    df.loc[df["co_applicant"]=="Not-Available","co_applicant_existing_emi"] = 0
    
    ##loan size
    print("calculating loan size probability...")
    df["loan_size"] = np.round(np.random.randint(5, 101, df.shape[0])) * 100000
    df["loan_duration"] = np.round(np.random.randint(3, 15, df.shape[0]))
    df = calculate_loan_size_probability(df)
    df["probability"] += df["loan_size_probability"]

    ##loan duration
    print("calculating loan duration probability...") 
    df = calculate_loan_duration_probability(df)
    df["probability"] += df["loan_duration_probability"]

    #date
    print("calculating date probability...")
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["probability"] = education_loan_opportunity(df)

    #categorical columns:
    print("calculating categorical probability...")
    df = categorical_probability(df, categories)


    ##total probability
    print("/n ***Probability Description***")
    print(f'''max {df["probability"].max()}, min {df["probability"].min()}, median {df["probability"].median()}, mean {df["probability"].mean()}''')
    
    #labels
    df["label"] = "Fair"  # Default label
    df.loc[(df["probability"] >= 0.7) & (df["co_applicant"]=="Not-Available"),"label"] = "Good"
    df.loc[(df["probability"] >= 0.7) & (df["co_applicant"]=="Available"),"label"] = "Good"
    df.loc[(df["probability"] <= 0.23) & (df["co_applicant"]=="Not-Available"),"label"] = "Not-Good"
    df.loc[(df["net_worth"] <= df['loan_size']) & (df["co_applicant"]=="Not-Available"),"label"] = "Not-Good"
    df.loc[(df["probability"] <= 0.3) & (df["co_applicant"]=="Available"),"label"] = "Not-Good"
    
    #dropping unnecessary columns
    df.drop(["age_group","applicant_networth_probability","rm_relation_probability","co_applicant_age_group",
             "co_applicant_networth_probability","co_applicant_salary_probability","loan_duration_probability",
             "loan_size_probability","eligible_emi","actual_loan_size","probability",
             "co_applicant_cibil_score_probability","cibil_score_probability","co_applicant_loan_history"],
            axis=1, inplace=True)

    #renaming the columns:
    df = df.rename({"age":"applicant_age",
                    "region":"applicant_region",
                    "marital_status":"applicant_marital_status",
                    "gender":"applicant_gender",
                    "education_level":"applicant_education_level",
                    "cibil":"applicant_cibil",
                    "net_worth":"applicant_networth"},axis=1)

    return df



