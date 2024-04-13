# standard
import pandas as pd

# machine learning tools
from sklearn.model_selection import train_test_split
import h2o
from h2o.estimators import H2OGradientBoostingEstimator

df = pd.read_csv('./data/osteoporosis.csv')
print(df.head(10))

# define features
features_num = ['Age']

features_cat = ['Gender', 'Hormonal Changes', 'Family History', 
                'Race/Ethnicity', 'Body Weight', 'Calcium Intake',
                'Vitamin D Intake', 'Physical Activity', 'Smoking',
                'Alcohol Consumption', 'Medical Conditions', 'Medications',
                'Prior Fractures']

target = 'Osteoporosis'

predictors = features_num + features_cat

# split data in train and test
df_train, df_test = train_test_split(df, test_size=None, random_state=4321)

# reset indices to avoid trouble when later adding predictions
df_train.reset_index(inplace=True)

# start H2O
h2o.init(max_mem_size='12G', nthreads=4) # Use maximum of 12 GB RAM and 4 cores

# upload data in H2O environment
train_hex = h2o.H2OFrame(df_train)

# force categorical target
train_hex[target] = train_hex[target].asfactor()

# define GBM
mono_config = {'Age':1} # force monotone increasing impact of age
gbm_model = H2OGradientBoostingEstimator(nfolds = 5,
                                         ntrees = 25,
                                         learn_rate = 0.05,
                                         max_depth = 4,
                                         min_rows = 10,
                                         col_sample_rate = 0.7,
                                         monotone_constraints = mono_config,
                                         score_each_iteration = True,
                                         stopping_rounds=5,
                                         stopping_metric='auc',
                                         stopping_tolerance=0.0001,
                                         seed=12345)

# train model
gbm_model.train(predictors, target, training_frame = train_hex)

test_data = {
	"Id": [391301844],
	"Age": [30],
	"Gender": ["Male"],
	"Hormonal Changes": ["Postmenopausal"],
	"Family History": ["Yes"],
	"Body Weight": ["Normal"],
	"Race/Ethnicity": ["Asian"],
	"Calcium Intake": ["Adequate"],
	"Vitamin D Intake": ["Insufficient"],
	"Physical Activity": ["Active"],
	"Smoking": ["Yes"],
	"Alcohol Consumption": ["None"],
	"Medical Conditions": ["Hyperthyroidism"],
	"Medications": ["Corticosteroids"],
	"Prior Fractures": ["Yes"]
}

def predictRisk(input_data):
	df_test = pd.DataFrame(input_data)
	pred = gbm_model.predict(h2o.H2OFrame(df_test))
	pred_list = h2o.as_list(pred)
	return pd.DataFrame.to_dict(pred_list, 'list')
