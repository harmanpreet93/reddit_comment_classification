import random
import pandas as pd


# return random classification
def random_classifier(classes):
	return classes[random.randint(0,len(classes)-1)]

def create_and_save_submission(predictions, file_name="submission.csv"):
	ids = [i for i in range(len(predictions))]
	sub_df = pd.DataFrame(data=list(zip(ids, predictions)), columns=["Id","Category"])
	sub_df.to_csv(file_name, index=False)

def main():
	train_data_path = "data/data_train.pkl"
	test_data_path = "data/data_test.pkl"

	# read train dataset
	train_data = pd.read_pickle(train_data_path)
	train_X = train_data[0]
	train_Y = train_data[1]
	classes = list(set(train_Y))

	# read test dataset
	test_data = pd.read_pickle(test_data_path)
	preds = []
	for _ in test_data:
		preds.append(random_classifier(classes))

	create_and_save_submission(preds, file_name="submission_random.csv")

if __name__ == '__main__':
	main()