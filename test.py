import numpy as np

import matplotlib.pyplot as plt

from stock_prediction import create_model, load_data
from parameters import *


def plot_graph(test_df):
    
    plt.plot(test_df[f'true_adjclose_{LOOKUP_STEP}'], c='b')
    plt.plot(test_df[f'adjclose_{LOOKUP_STEP}'], c='r')
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend(["Actual Price", "Predicted Price"])
    plt.show()


def get_final_df(model, data):
    
    buy_profit  = lambda current, pred_future, true_future: true_future - current if pred_future > current else 0
    
    
    sell_profit = lambda current, pred_future, true_future: current - true_future if pred_future < current else 0
    X_test = data["X_test"]
    y_test = data["y_test"]
    
    y_pred = model.predict(X_test)
    if SCALE:
        y_test = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(np.expand_dims(y_test, axis=0)))
        y_pred = np.squeeze(data["column_scaler"]["adjclose"].inverse_transform(y_pred))
    test_df = data["test_df"]
    
    test_df[f"adjclose_{LOOKUP_STEP}"] = y_pred
    
    test_df[f"true_adjclose_{LOOKUP_STEP}"] = y_test
    
    test_df.sort_index(inplace=True)
    final_df = test_df
    
    final_df["buy_profit"] = list(map(buy_profit,
                                    final_df["adjclose"],
                                    final_df[f"adjclose_{LOOKUP_STEP}"],
                                    final_df[f"true_adjclose_{LOOKUP_STEP}"])
                                    
                                    )
    
    final_df["sell_profit"] = list(map(sell_profit,
                                    final_df["adjclose"],
                                    final_df[f"adjclose_{LOOKUP_STEP}"],
                                    final_df[f"true_adjclose_{LOOKUP_STEP}"])
                                    
                                    )
    return final_df


def predict(model, data):
    
    last_sequence = data["last_sequence"][-N_STEPS:]
    
    last_sequence = np.expand_dims(last_sequence, axis=0)
    
    prediction = model.predict(last_sequence)
    
    if SCALE:
        predicted_price = data["column_scaler"]["adjclose"].inverse_transform(prediction)[0][0]
    else:
        predicted_price = prediction[0][0]
    return predicted_price



data = load_data(ticker, N_STEPS, scale=SCALE, split_by_date=SPLIT_BY_DATE,
                shuffle=SHUFFLE, lookup_step=LOOKUP_STEP, test_size=TEST_SIZE,
                feature_columns=FEATURE_COLUMNS)


model = create_model(N_STEPS, len(FEATURE_COLUMNS), loss=LOSS, units=UNITS, cell=CELL, n_layers=N_LAYERS,
                    dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)


model_path = os.path.join("results", model_name) + ".h5"
model.load_weights(model_path)


loss, mae = model.evaluate(data["X_test"], data["y_test"], verbose=0)

if SCALE:
    mean_absolute_error = data["column_scaler"]["adjclose"].inverse_transform([[mae]])[0][0]
else:
    mean_absolute_error = mae


final_df = get_final_df(model, data)

future_price = predict(model, data)

accuracy_score = (len(final_df[final_df['sell_profit'] > 0]) + len(final_df[final_df['buy_profit'] > 0])) / len(final_df)

total_buy_profit  = final_df["buy_profit"].sum()
total_sell_profit = final_df["sell_profit"].sum()

total_profit = total_buy_profit + total_sell_profit

profit_per_trade = total_profit / len(final_df)

print(f"Future price after {LOOKUP_STEP} days is {future_price:.2f}$")
print(f"{LOSS} loss:", loss)
print("Mean Absolute Error:", mean_absolute_error)
print("Accuracy score:", accuracy_score)
#print("Total buy profit:", total_buy_profit)
#print("Total sell profit:", total_sell_profit)
#print("Total profit:", total_profit)
#print("Profit per trade:", profit_per_trade)
# plot true/pred prices graph
plot_graph(final_df)
print(final_df.tail(10))

csv_results_folder = "csv-results"
if not os.path.isdir(csv_results_folder):
    os.mkdir(csv_results_folder)
csv_filename = os.path.join(csv_results_folder, model_name + ".csv")
final_df.to_csv(csv_filename)
