import pandas as pd
import numpy as np
import os
import tensorflow as tf

#from utils import build_vocab_files, show_group_stats_viz, aggregate_dataset, preprocess_df, df_to_dataset, posterior_mean_field, prior_trainable

####### STUDENTS FILL THIS OUT ######
#Question 3
def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''
    df = df.merge(ndc_df[['NDC_Code','Non-proprietary Name']], left_on='ndc_code', right_on='NDC_Code')
    df.rename(columns={'Non-proprietary Name': 'generic_drug_name'}, inplace=True)
    return df

#Question 4
def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return: df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''
    df.head()
    selection = df.groupby('patient_nbr').agg({'encounter_id':'min'}).encounter_id.values
    df = df.loc[df.encounter_id.isin(selection)]
    return df


#Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    idxs = df[patient_key].unique()
    train_idx = np.random.choice(idxs, int(0.6 * len(idxs)))
    tmp_idx = set(idxs) - set(train_idx)
    val_idx = np.random.choice(list(tmp_idx), int(0.5 * len(idxs)))
    test_idx = list(set(tmp_idx) - set(val_idx))
    train = df.loc[df.patient_nbr.isin(train_idx)]
    validation = df.loc[df.patient_nbr.isin(val_idx)]
    test = df.loc[df.patient_nbr.isin(test_idx)]
    return train, validation, test

#Question 7

def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        size = pd.read_csv(vocab_file_path, header=None).shape[0]
        tf_categorical_column = tf.feature_column.categorical_column_with_vocabulary_file(key=c, vocabulary_file=vocab_file_path, vocabulary_size= size)
        #tf_categorical_feature_column = tf.feature_column.indicator_column(tf_categorical_column)
        tf_categorical_feature_column = tf.feature_column.embedding_column(tf_categorical_column, max(2,size//10))
        output_tf_list.append(tf_categorical_feature_column)
    return output_tf_list

#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std



def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field
    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    tf_numeric_feature = tf.feature_column.numeric_column(col, shape=(1,), default_value=default_value, normalizer_fn=lambda x: normalize_numeric_with_zscore(x, MEAN, STD))
    return tf_numeric_feature

#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = '?'
    s = '?'
    return m, s

# Question 10
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    return student_binary_prediction


if __name__ == '__main__':
    def select_model_features(df, categorical_col_list, numerical_col_list, PREDICTOR_FIELD,
                              grouping_key='patient_nbr'):
        selected_col_list = [grouping_key] + [PREDICTOR_FIELD] + categorical_col_list + numerical_col_list
        return agg_drug_df[selected_col_list]


    def aggregate_dataset(df, grouping_field_list, array_field):
        df = df.groupby(grouping_field_list)['encounter_id',
                                             array_field].apply(
            lambda x: x[array_field].values.tolist()).reset_index().rename(columns={
            0: array_field + "_array"})

        dummy_df = pd.get_dummies(df[array_field + '_array'].apply(pd.Series).stack()).sum(level=0)
        dummy_col_list = [x.replace(" ", "_") for x in list(dummy_df.columns)]
        mapping_name_dict = dict(zip([x for x in list(dummy_df.columns)], dummy_col_list))
        concat_df = pd.concat([df, dummy_df], axis=1)
        new_col_list = [x.replace(" ", "_") for x in list(concat_df.columns)]
        concat_df.columns = new_col_list

        return concat_df, dummy_col_list


    def cast_df(df, col, d_type=str):
        return df[col].astype(d_type)

    def impute_df(df, col, impute_value=0):
        return df[col].fillna(impute_value)

    def preprocess_df(df, categorical_col_list, numerical_col_list, predictor, categorical_impute_value='nan', numerical_impute_value=0):
        df[predictor] = df[predictor].astype(float)
        for c in categorical_col_list:
            df[c] = cast_df(df, c, d_type=str)
        for numerical_column in numerical_col_list:
            df[numerical_column] = impute_df(df, numerical_column, numerical_impute_value)
        return df


    dataset_path = "./data/final_project_dataset.csv"
    df = pd.read_csv(dataset_path)

    ndc_code_path = "./medication_lookup_tables/final_ndc_lookup_table"
    ndc_code_df = pd.read_csv(ndc_code_path)

    reduce_dim_df = reduce_dimension_ndc(df, ndc_code_df)
    first_encounter_df = select_first_encounter(reduce_dim_df)

    exclusion_list = ['generic_drug_name', 'ndc_code', 'NDC_Code']
    grouping_field_list = [c for c in first_encounter_df.columns if c not in exclusion_list]
    agg_drug_df, ndc_col_list = aggregate_dataset(first_encounter_df, grouping_field_list, 'generic_drug_name')

    assert len(agg_drug_df) == agg_drug_df['patient_nbr'].nunique() == agg_drug_df['encounter_id'].nunique()

    required_demo_col_list = ['race', 'gender', 'age']
    student_categorical_col_list = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id',
                                    'medical_specialty', 'primary_diagnosis_code',
                                    'other_diagnosis_codes', 'max_glu_serum', 'A1Cresult',
                                    'change', ] + required_demo_col_list + ndc_col_list
    student_numerical_col_list = numericals = ['time_in_hospital', 'number_outpatient', 'number_inpatient',
                                               'number_emergency',
                                               'num_lab_procedures', 'number_diagnoses', 'num_medications',
                                               'num_procedures']
    PREDICTOR_FIELD = 'readmitted'
    selected_features_df = select_model_features(agg_drug_df, student_categorical_col_list, student_numerical_col_list, PREDICTOR_FIELD)
    map_target = dict({'NO': 0, '>30': 1, '<30': 2})
    selected_features_df[PREDICTOR_FIELD] = selected_features_df[PREDICTOR_FIELD].apply(lambda x: map_target[x])
    processed_df = preprocess_df(selected_features_df, student_categorical_col_list, student_numerical_col_list, PREDICTOR_FIELD,
                                 categorical_impute_value='nan', numerical_impute_value=0)

    d_train, d_val, d_test = patient_dataset_splitter(processed_df, 'patient_nbr')