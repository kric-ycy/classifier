
### filter the obect columns from the raw data
import pandas as pd

class ExcelProcessor:
    def __init__(self):
        pass


    @staticmethod
    def get_object_columns(df):
        """
        Get columns with object data type from a DataFrame.
        
        Parameters:
        df (pd.DataFrame): The DataFrame to analyze.
        
        Returns:
        list: A list of column names with object data type.
        """
        df_part = df.iloc[3:]
        object_columns = [col for col in df_part.columns if df_part[col].dropna().dtype == 'object']
        filtered_columns = [col for col in object_columns if col.startswith("Q") or col.startswith("sq")]
        return filtered_columns

    @staticmethod
    def get_question_columns(df, q_num , grouped_dict):
        columns = grouped_dict.get(q_num  , [])
        r_df = df[columns]
        return r_df

    def read_excel(self, file_path, sheet_name='DATA'):

        raw = pd.read_excel(file_path, sheet_name=sheet_name)
        raw = raw.set_index("SEQ")
        object_columns = self.get_object_columns(raw)
        obj_data = raw[object_columns].iloc[2:]
        q_numbers = list(set([question.replace("z","_").split("_")[0] for question in object_columns]))
        
        grouped_dict = {prefix: [] for prefix in q_numbers}
        
        for q_num in q_numbers:
            grouped_columns = [col for col in obj_data.columns if col.startswith(q_num)]
            grouped_dict[q_num] = grouped_columns

        return raw, obj_data, grouped_dict





# df = get_question_columns(obj_data, "Q5", grouped_dict).dropna()
