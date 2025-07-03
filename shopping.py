import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    evidence = []
    labels = []
    with open(filename) as f:
        file = csv.reader(f)
        next(file)

    
        month_to_number = {
            "Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3,
            "May": 4, "June": 5, "Jul": 6, "Aug": 7,
            "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11
        }

        visitor_type = {
            "Returning_Visitor": 1,
            "New_Visitor": 0,
            "Other": 0
        }

        weekend = {
            "TRUE": 1,
            "FALSE": 0
        }
        revenue = {
            "TRUE": 1, 
            "FALSE": 0
        }

        data = []
        for row in file:
            data.append({
                "evidence": [
                    int(row[num]) if num == 0 
                    else float(row[num]) if num == 1
                    else int(row[num]) if num == 2
                    else float(row[num]) if num == 3
                    else int(row[num]) if num == 4
                    else float(row[num]) if num == 5
                    else float(row[num]) if num == 6
                    else float(row[num]) if num == 7
                    else float(row[num]) if num == 8
                    else float(row[num]) if num == 9
                    else month_to_number[row[num]] if num == 10
                    else int(row[num]) if num == 11
                    else int(row[num]) if num == 12
                    else int(row[num]) if num == 13
                    else int(row[num]) if num == 14
                    else visitor_type[row[num]] if num == 15
                    else weekend[row[num]] if num == 16
                    else None
                    for num in range(0, len(row)-1)
                    ],
                "label": revenue[row[-1]]
            })
    evidence = [row["evidence"] for row in data]
    labels = [row["label"] for row in data]
    return (evidence, labels)
    


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    predicted_purchases = 0
    predicted_nonpurchases = 0
    number_of_purchases = 0
    number_of_nonpurchases = 0
    
    # Count the number of purchases and the number of non purchases
    for i in range(len(labels)):
        if labels[i] == 1:
            number_of_purchases += 1
        else:
            number_of_nonpurchases += 1

        # Count the number of correctly predicted purchases and number of correct ignored non purchases
        if labels[i] == 1 and predictions[i] == 1:
            predicted_purchases += 1
        if labels[i] == 0 and predictions[i] == 0:
            predicted_nonpurchases += 1

    # Return the sensitivity and specificity
    return ((float(predicted_purchases/number_of_purchases), float(predicted_nonpurchases/number_of_nonpurchases)))



if __name__ == "__main__":
    main()
