import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from tkinter.ttk import Progressbar
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C, DQN, PPO, DDPG, HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.buffers import ReplayBuffer
import pandas as pd
from env_fx import tgym
import os
import gym
import MetaTrader5 as mt5
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import pytz
from datetime import datetime, date, timedelta
import random
import math
import numpy as np
from DataProcessor import DataProcessor
import pickle
from Data_train.DataCollector import DataCollector


window = tk.Tk()
window.title("RL Training GUI")

position = random.randint(100, 5000)
DataProc = DataProcessor(realtime=True)


def display_correlation_matrix(correlation_matrix_tab, data_for_all_symbols):
    # Create a correlation matrix from the data
    symbols = data_for_all_symbols["Symbol_Decoded"]
    prices = data_for_all_symbols[["Open", "High", "Low", "Close"]].values

    correlation_matrix = np.corrcoef(prices)

    # Check if correlation_matrix is empty or has an unexpected shape
    if correlation_matrix.size == 0 or correlation_matrix.ndim != 2:
        print("Invalid shape or empty correlation matrix:",
              correlation_matrix.shape)
        return

    # Create the figure and subplot
    fig = plt.figure(figsize=(8, 6))
    correlation_subplot = fig.add_subplot(111)

    # Plot the correlation matrix as a heatmap
    heatmap = correlation_subplot.imshow(
        correlation_matrix, cmap='RdYlGn', interpolation='nearest')
    # Add text annotations to the heatmap
    for i in range(len(symbols)):
        for j in range(len(symbols)):
            text = correlation_subplot.text(j, i, f"{correlation_matrix[i, j]:.2f}",
                                            ha='center', va='center', color='black')
    correlation_subplot.set_xticks(np.arange(len(symbols)))
    correlation_subplot.set_yticks(np.arange(len(symbols)))
    correlation_subplot.grid(
        which='minor', color='black', linestyle='-', linewidth=1)
    correlation_subplot.set_xticklabels(symbols, rotation=45)
    correlation_subplot.set_yticklabels(symbols)
    fig.colorbar(heatmap)

    # Set the title for the subplot
    correlation_subplot.set_title("Correlation Matrix")

    # Create a canvas to display the figure in the tab
    canvas = FigureCanvasTkAgg(fig, master=correlation_matrix_tab)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


def get_info(symbol):
    '''https://www.mql5.com/en/docs/integration/python_metatrader5/mt5symbolinfo_py
    '''
    # get symbol properties
    info = mt5.symbol_info(symbol)
    return info


def close_trade(action, buy_request, result, deviation):
    '''https://www.mql5.com/en/docs/integration/python_metatrader5/mt5ordersend_py
    '''
    # create a close request
    symbol = buy_request['symbol']
    if action == 'buy':
        trade_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol).ask
    elif action == 'sell':
        trade_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).bid
    position_id = result.order
    lot = buy_request['volume']

    close_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_SELL,
        "position": position_id,
        "price": price,
        "deviation": deviation,
        "magic": 9986989,
        "comment": "python script close",
        "type_time": mt5.ORDER_TIME_GTC,  # good till cancelled
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    # send a close request
    result = mt5.order_send(close_request)


def open_trade(action, symbol, lot, sl_points, tp_points, deviation=0):
    '''https://www.mql5.com/en/docs/integration/python_metatrader5/mt5ordersend_py
    '''
    # prepare the buy request structure
    symbol_info = get_info(symbol)

    if action == 'buy':
        trade_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol).ask
    elif action == 'sell':
        trade_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(symbol).bid
    point = mt5.symbol_info(symbol).point

    buy_request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": trade_type,
        "price": price,
        "sl": price - sl_points * point,
        "tp": price + tp_points * point,
        "deviation": deviation,
        "magic": 9986989,
        "comment": "sent by python",
        "type_time": mt5.ORDER_TIME_GTC,  # good till cancelled
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    # send a trading request
    result = mt5.order_send(buy_request)

    return result, buy_request


def display_graphs(graphs_tab, data):
    # Extract currency pairs and exchange rates
    currency_pairs = data["Symbol"]
    exchange_rates = data['Close']

    # Create the figure and subplots
    fig = Figure(figsize=(8, 6))
    subplot = fig.add_subplot(111)

    # Plot the lines
    subplot.plot(currency_pairs, exchange_rates, marker='o')

    # Set the x-axis label
    subplot.set_xlabel('Currency Pairs')

    # Set the y-axis label
    subplot.set_ylabel('Exchange Rates')

    # Set the title
    subplot.set_title('Exchange Rates for Currency Pairs')

    # Rotate x-axis labels for better visibility
    subplot.tick_params(axis='x', rotation=45)

    # Create a canvas to display the figure in the tab
    canvas = FigureCanvasTkAgg(fig, master=graphs_tab)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


def train_model(ea_strategy, algorithm_name, my_timesteps, save_freq, model_directory, chckpcallback, myoption, env,
                progressbar, max_timesteps):
    data_df = pd.read_csv("Data_train/MT5data.csv")
    seed = 23

    # Creating both directories
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    log_directory = 'logs'  # defining the log directory for tensorboard
    buffer_directory = 'ReplayBuffer'

    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    if not os.path.exists(buffer_directory):
        os.makedirs(buffer_directory)

    # This function chooses whether to load an existing model for further training or to train a new model

    def instantiate_model(myoption, env, log_directory, seed):
        if myoption == "New Training":  # Option for training a new model
            model = eval(
                '{}(\'MlpPolicy\', env, verbose=1, replay_buffer_class=ReplayBuffer, tensorboard_log=log_directory, '
                'learning_rate=0.0002, gamma=0.98, device="cuda", buffer_size=1000000, seed=seed)'.format(
                    algorithm_name))
        elif myoption == "Resume Training":  # Option for resuming training from a checkpoint of your choice
            # Loading model from the directory path
            my_path = simpledialog.askinteger("Previous Checkpoint",
                                              "Enter the number of steps of the previous checkpoint:",
                                              parent=window, minvalue=0)
            dir_path = 'Check_points_{}/{}_RL_{}_steps.zip'.format(
                algorithm_name, algorithm_name, str(my_path))
            model = eval('{}.load(dir_path, env=env)'.format(algorithm_name))
            model.load_replay_buffer('Check_points_{}/{}_RL_replay_buffer_{}_steps.pkl'.format(algorithm_name,
                                                                                               algorithm_name,
                                                                                               str(my_path)))
            print(
                f"\nThe loaded model has {model.replay_buffer.size()} transitions in its buffer")
        return model

    model = instantiate_model(myoption, env, log_directory, seed)
    range_var = math.ceil(max_timesteps/my_timesteps)
    for i in range(1, range_var):
        # Model starts learning using this
        model.learn(total_timesteps=my_timesteps, callback=chckpcallback, reset_num_timesteps=False,
                    tb_log_name=algorithm_name)
        model.save("{}/{}".format(model_directory, my_timesteps * i))
        progressbar["value"] = (i / range_var) * 100
        progress = tk.Label(window, text="Current Progress: {}%".format(
            round(progressbar['value'], 1)))
        progress.grid(row=7, column=0, columnspan=2, pady=10)
        window.update()


def my_callback(algorithm_name, saving_freq):
    chckpcallback = CheckpointCallback(save_freq=saving_freq, save_path='Check_points_{}'.format(algorithm_name),
                                       name_prefix='{}_RL'.format(algorithm_name), save_replay_buffer=True)
    return chckpcallback


def deploy_trained_model():
    iterator = 0
    with open('assets.pkl', 'rb') as f:
        assets = pickle.load(f)
    assets_env=assets
    data_for_all_symbols = pd.DataFrame(columns=[
                                        'Open', 'High', 'Low', 'Close', 'Symbol', 'Top_Gainer', 'Top_Loser', 'Percentage Change', 'Hit Time', 'Hit Price'])
    current_date = date.today()
    timezone = pytz.timezone("Etc/UTC")
    open_prices_today = {}

    top_gain_symbol = []
    top_loss_symbol = []

    start_time = datetime(
        current_date.year, current_date.month, current_date.day, 0, 0, tzinfo=timezone)

    for symbol_name in assets_env:
        symbol_data_day = mt5.copy_rates_from(
            symbol_name, mt5.TIMEFRAME_D1, start_time, 1)

        open_prices_today[symbol_name] = symbol_data_day[0][1]

    while iterator < len(assets_env):
        symbol_data = mt5.copy_rates_from_pos(
            assets_env[iterator], mt5.TIMEFRAME_M1, 0, 1)

        if symbol_data is not None:
            symbol = assets_env[iterator]
            # Check if the symbol already exists in the DataFrame
            if symbol not in data_for_all_symbols['Symbol'].values:
                # Extract the desired data from symbol_data and create a row to append
                current_time = datetime.fromtimestamp(symbol_data[0][0])
                current_time = current_time.astimezone(pytz.utc)
                spread = symbol_data[0][-2]
                row = {
                    'Open': float(symbol_data[0][1]),
                    'High': float(symbol_data[0][2]),
                    'Low': float(symbol_data[0][3]),
                    'Close': float(symbol_data[0][4]),
                    'Symbol': symbol,
                    'Top_Gainer': '',
                    'Top_Loser': '',
                    'Percentage Change': '',
                    'Hit Time': current_time,
                    'Hit Price': float(symbol_data[0][4]),
                    'Spread': spread
                }
                # Append the row to the DataFrame
                data_for_all_symbols = data_for_all_symbols.append(
                    row, ignore_index=True)

            iterator += 1

    # Remove duplicate symbols
    data_for_all_symbols.drop_duplicates(
        subset='Symbol', keep='first', inplace=True)

    data_for_all_symbols = DataProc.Calculate_percentage_change(
        data_for_all_symbols, open_prices_today, assets_env)
    data_for_all_symbols = DataProc.Calculate_top_Gainers(data_for_all_symbols)
    data_for_all_symbols = DataProc.Calculate_top_Losers(data_for_all_symbols)
    data_for_all_symbols = DataProc.Encode_symbols(data_for_all_symbols)


    # Fetch the top gainers and top losers data from your DataFrame
    top_gainers = data_for_all_symbols[data_for_all_symbols['Top_Gainer'] == True]
    top_losers = data_for_all_symbols[data_for_all_symbols['Top_Loser'] == True]

    # Prepare the data in the required format
    top_gain_symbol.append((top_gainers['Symbol'].values))
    top_loss_symbol.append((top_losers['Symbol'].values))

    result, buy_request = open_trade('buy', 'AUDCAD', 0.1, 50, 50)
    #close_trade('sell', buy_request, result, 10)

    return data_for_all_symbols


def deploy_GUI(leverage,balance,equity,ai_filter,algo_name):
   
    if ai_filter == 'AI Trade Filter':
        ai_filter=True
    else:
        ai_filter=False
        
    algo_class = eval(f"{algo_name}")
    # Instantiate the algorithm class
    algorithm = algo_class.load('models\\DDPG\\2000.zip')
        
    
    equity_env=equity

    def close_trades():
        # Implement the logic to close all trades in MetaTrader 5
        pass

    

    data_for_all_symbols = deploy_trained_model()
    
    def step_rl_agent():
        account_info = mt5.account_info()

        leverage = account_info[2]
        balance = account_info[10]
        equity = account_info[13]
        env=tgym(leverage,balance,equity,ai_filter,training=False,df=data_for_all_symbols)
        obs=env.reset()
        action=algorithm.predict(obs)
        print(action)

    deploy_window = tk.Toplevel(window)
    deploy_window.title("Deploy Model")
    deploy_window.geometry('400x300')

    notebook = ttk.Notebook(deploy_window)

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": "AUDCAD",
        "volume": 3,
        "type": mt5.ORDER_TYPE_BUY,
        "price": 105,
        "sl": 105 - 100 * 1,
        "tp": 105 + 100 * 1,
        "deviation": 0.4,
        "magic": 234000,
        "comment": "python script open",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }

    result = mt5.order_send(request)

    def stop_trading():
        # Enable entries
        max_spread_entry.configure(state="normal")
        max_lots_entry.configure(state="normal")
        max_profit_entry.configure(state="normal")
        max_drawdown_entry.configure(state="normal")

        # Enable comboboxes
        ai_filter_combobox.configure(state="normal")
        start_ea_combobox.configure(state="normal")
        max_spread_combobox.configure(state="normal")
        max_lots_combobox.configure(state="normal")
        max_profit_combobox.configure(state="normal")
        max_drawdown_combobox.configure(state="normal")
        start_trading_button.configure(state="normal")
        stop_trading_button.configure(state="disabled")

    def start_trading():
        # Disable entries
        max_spread_entry.configure(state="disabled")
        max_lots_entry.configure(state="disabled")
        max_profit_entry.configure(state="disabled")
        max_drawdown_entry.configure(state="disabled")

        # Disable comboboxes
        ai_filter_combobox.configure(state="disabled")
        start_ea_combobox.configure(state="disabled")
        max_spread_combobox.configure(state="disabled")
        max_lots_combobox.configure(state="disabled")
        max_profit_combobox.configure(state="disabled")
        max_drawdown_combobox.configure(state="disabled")
        start_trading_button.configure(state='disabled')
        stop_trading_button.configure(state="normal")
        step_rl_agent()
    # Create the dashboard indicator tab
    dashboard_tab = ttk.Frame(notebook)
    graph_tab = ttk.Frame(notebook)
    correlation_matrix = ttk.Frame(notebook)
    Top_gainers_coloured = ttk.Frame(notebook)

    ai_pos_sz = round((position/equity_env)*100, 2)

    ai_position_label = ttk.Label(
        dashboard_tab, text="AI Position Size: {}%".format(ai_pos_sz))
    total_lots_label = ttk.Label(
        dashboard_tab, text="Total Lots Open: {}".format(0))
    total_trades_label = ttk.Label(
        dashboard_tab, text="Total Trades Open: {}".format(0))
    total_pnl_label = ttk.Label(
        dashboard_tab, text="Total Pips/Points PnL: {}".format(3))
    total_percent_pnl_label = ttk.Label(dashboard_tab, text="Total %PnL:")

    close_trades_button = ttk.Button(dashboard_tab, text="Close All Trades")
    start_trading_button = ttk.Button(
        dashboard_tab, text="Start Trading", command=start_trading)
    stop_trading_button = ttk.Button(
        dashboard_tab, text="Stop Trading", command=stop_trading)
    stop_trading_button.configure(state="disabled")

    # Grid layout for dashboard indicators
    ai_position_label.grid(row=0, column=0, sticky=tk.W)
    total_lots_label.grid(row=1, column=0, sticky=tk.W)
    total_trades_label.grid(row=2, column=0, sticky=tk.W)
    total_pnl_label.grid(row=3, column=0, sticky=tk.W)
    total_percent_pnl_label.grid(row=4, column=0, sticky=tk.W)
    close_trades_button.grid(row=14, column=0, pady=10, sticky=tk.W)
    start_trading_button.grid(row=14, column=1, sticky=tk.W)
    stop_trading_button.grid(row=14, column=2, sticky=tk.W)

    ai_filter_label = ttk.Label(dashboard_tab, text="Ai Trade Filter: ")
    ai_filter_combobox = ttk.Combobox(
        dashboard_tab, values=["On", "Off"], width=5, justify='center')

    start_ea_label = ttk.Label(
        dashboard_tab, text="Start EA at Next Daily Open: ")
    start_ea_combobox = ttk.Combobox(
        dashboard_tab, values=["On", "Off"], width=5, justify='center')

    max_spread_label = ttk.Label(dashboard_tab, text="Max Spread: ")
    max_spread_combobox = ttk.Combobox(
        dashboard_tab, values=["On", "Off"], width=5, justify='center')

    max_lots_label = ttk.Label(dashboard_tab, text="Max Lots Per Trade: ")
    max_lots_combobox = ttk.Combobox(
        dashboard_tab, values=["On", "Off"], width=5,  justify='center')

    max_profit_label = ttk.Label(dashboard_tab, text="Max Daily Profit %: ")
    max_profit_combobox = ttk.Combobox(
        dashboard_tab, values=["On", "Off"], width=5, justify='center')

    max_drawdown_label = ttk.Label(
        dashboard_tab, text="Max Daily DrawDown %: ")
    max_drawdown_combobox = ttk.Combobox(
        dashboard_tab, values=["On", "Off"], width=5, justify='center')

    # Entry widgets for input values
    max_spread_entry = ttk.Entry(dashboard_tab)
    max_lots_entry = ttk.Entry(dashboard_tab)
    max_profit_entry = ttk.Entry(dashboard_tab)
    max_drawdown_entry = ttk.Entry(dashboard_tab)

    ai_filter_label.grid(row=7, column=0, sticky=tk.W)
    ai_filter_combobox.grid(row=7, column=1, sticky=tk.W)

    start_ea_label.grid(row=8, column=0, sticky=tk.W)
    start_ea_combobox.grid(row=8, column=1, sticky=tk.W)

    max_spread_label.grid(row=9, column=0, sticky=tk.W)
    max_spread_combobox.grid(row=9, column=1, sticky=tk.W)
    max_spread_entry.grid(row=9, column=2, sticky=tk.W)

    max_lots_label.grid(row=10, column=0, sticky=tk.W)
    max_lots_combobox.grid(row=10, column=1, sticky=tk.W)
    max_lots_entry.grid(row=10, column=2, sticky=tk.W)

    max_profit_label.grid(row=11, column=0, sticky=tk.W)
    max_profit_combobox.grid(row=11, column=1, sticky=tk.W)
    max_profit_entry.grid(row=11, column=2, sticky=tk.W)

    max_drawdown_label.grid(row=12, column=0, sticky=tk.W)
    max_drawdown_combobox.grid(row=12, column=1, sticky=tk.W)
    max_drawdown_entry.grid(row=12, column=2, sticky=tk.W)

    # Add tabs to the notebook
    notebook.add(dashboard_tab, text="Auto-Trader Dashboard")
    #notebook.add(graph_tab, text="Charts Display")
    notebook.add(correlation_matrix, text="Correlation Matrix")
    notebook.add(Top_gainers_coloured, text="Top Gainers/Losers")


    display_graphs(graph_tab, data_for_all_symbols)
    display_correlation_matrix(correlation_matrix, data_for_all_symbols)

    sorted_symbols = data_for_all_symbols.sort_values(
        by="Percentage Change", ascending=False)

    # Create the treeview widget
    top_gainers_losers_table = ttk.Treeview(
        Top_gainers_coloured, columns=("Percentage Change",), show="tree headings")
    top_gainers_losers_table.heading("#0", text="GAINERS/LOSERS")
    top_gainers_losers_table.heading(
        "Percentage Change", text="Percentage Change")

    # Configure column widths
    top_gainers_losers_table.column("#0", width=100)
    top_gainers_losers_table.column("Percentage Change", width=100)

    # Configure the style to enable gridlines
    style = ttk.Style()
    style.configure("Custom.Treeview",
                    rowheight=25,
                    fieldbackground="white",
                    borderwidth=1,
                    relief="solid",
                    show="tree headings")  # Show the tree headings
    style.configure("Custom.Treeview.Heading", font=(
        "TkDefaultFont", 10, "bold"))  # Customize the heading font

    # Configure tags for coloring cells
    style.configure("gainer.Treeview", background="green")
    style.configure("loser.Treeview", background="red")

    # Apply the style to the treeview
    top_gainers_losers_table.configure(style="Custom.Treeview")

    # Add symbols and percentage change to the treeview and color cells
    for symbol, percentage_change in zip(sorted_symbols["Symbol_Decoded"], sorted_symbols["Percentage Change"]):
        tags = ()
        if sorted_symbols.loc[sorted_symbols['Symbol_Decoded'] == symbol, 'Top_Gainer'].item():
            tags = ("gainer",)
        elif sorted_symbols.loc[sorted_symbols['Symbol_Decoded'] == symbol, 'Top_Loser'].item():
            tags = ("loser",)
        top_gainers_losers_table.insert("", "end", text=symbol, values=(
            f"{round(percentage_change, 2)}%",), tags=tags)

    # Configure tags for coloring cells
    top_gainers_losers_table.tag_configure("gainer", background="green")
    top_gainers_losers_table.tag_configure("loser", background="red")

    # Configure the style to enable gridlines
    style = ttk.Style()
    style.configure("Custom.Treeview", rowheight=25,
                    fieldbackground="white", borderwidth=5, relief="solid")
    # Configure background color for selected rows
    style.map("Custom.Treeview", background=[("selected", "#ececec")])
    top_gainers_losers_table.configure(style="Custom.Treeview")

    # Grid layout for the treeview widget
    top_gainers_losers_table.grid(row=0, column=0, sticky="nsew")
    Top_gainers_coloured.grid_rowconfigure(0, weight=1)
    Top_gainers_coloured.grid_columnconfigure(0, weight=1)

    # Pack the notebook widget
    notebook.pack(expand=True, fill=tk.BOTH)


def login_gui():
    # Check if login information file exists
    login_info_file = 'login_info.pickle'
    if os.path.isfile(login_info_file):
        with open(login_info_file, 'rb') as file:
            login_info = pickle.load(file)
            saved_account = login_info['account']
            saved_password = login_info['password']

            save_login = tk.BooleanVar(value=True)
    else:
        saved_account = ''
        saved_password = ''
        save_login = tk.BooleanVar()

    def login_button_click():
        account = account_entry.get()
        password = password_entry.get()
        save_login_info = save_login.get()  # Check if "Keep me logged in" is checked

        # Connect to your MetaTrader 5 account
        mt5.initialize()
        authorized = mt5.login(account, password)

        # Save login information if "Keep me logged in" is checked
        if save_login_info:
            login_info = {'account': account, 'password': password}
            with open(login_info_file, 'wb') as file:
                pickle.dump(login_info, file)

        else:
            if os.path.isfile(login_info_file):
                os.remove(login_info_file)

        account_info = mt5.account_info()

        leverage = account_info[2]
        balance = account_info[10]
        equity = account_info[13]

        

        if not authorized:
            messagebox.showinfo("Login Successful",
                                "You have successfully logged in!")
            login_window.withdraw()
            train_model_gui(leverage=leverage,balance=balance,equity=equity)

    login_window = tk.Toplevel(window)
    login_window.title("LOGIN DETAILS")

    notebook = ttk.Notebook(login_window)

    login_tab = ttk.Frame(notebook)

    notebook.add(login_tab, text="Login")
    notebook.pack(expand=True, fill=tk.BOTH)

    # Create the account number label and entry field
    account_label = ttk.Label(login_tab, text="Account Number:")
    account_entry = ttk.Entry(login_tab)

    # Create the password label and entry field
    password_label = ttk.Label(login_tab, text="Password:")
    password_entry = ttk.Entry(login_tab, show="*")

    # Create the login button
    login_button = ttk.Button(login_tab, text="Login",
                              command=login_button_click)

    save_login_checkbox = ttk.Checkbutton(
        login_tab, text="Keep me logged in", variable=save_login)

    # Grid layout for the widgets
    account_label.grid(row=0, column=0, sticky=tk.W)
    account_entry.grid(row=0, column=1)
    password_label.grid(row=1, column=0, sticky=tk.W)
    password_entry.grid(row=1, column=1)
    save_login_checkbox.grid(row=3, column=0, columnspan=2)

    login_button.grid(row=2, column=0, columnspan=2, pady=10)

    account_entry.insert(tk.END, saved_account)
    password_entry.insert(tk.END, saved_password)

    window.mainloop()

    return True


def data_collection_gui():
    data_collection_window = tk.Toplevel()
    data_collection_window.title("Data Collection GUI")

    # Create input fields for start date and end date
    start_date_label = tk.Label(data_collection_window, text="Start Date (YYYY-MM-DD):")
    start_date_entry = tk.Entry(data_collection_window)
    end_date_label = tk.Label(data_collection_window, text="End Date (YYYY-MM-DD):")
    end_date_entry = tk.Entry(data_collection_window)

    
    
    

    # Create a dropdown menu for time frame
    time_frame_label = tk.Label(data_collection_window, text="Time Frame:")
    time_frame_var = tk.StringVar()
    time_frame_dropdown = ttk.Combobox(data_collection_window, textvariable=time_frame_var)
    time_frame_dropdown['values'] = ['M1','M5','M15','M30','H1','H4','D1','W1','MN1']
    time_frame_dropdown.current(0)

    # Function to handle the "Continue Collection" button click event
    def continue_collection_click():
        start_date_str = start_date_entry.get()
        end_date_str = end_date_entry.get()
        timeframe_str=time_frame_var.get()
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
        collection_obj=DataCollector(start_date, end_date,timeframe_str, cont=True)
        collection_obj.start_collection()
        progress=collection_obj.progress

        data_collection_window.destroy()

    # Function to handle the "Collect New" button click event
    def collect_new_click():
        # Implement your collect new logic here
        # ...

        # Get the start and end dates from the input fields
        start_date_str = start_date_entry.get()
        end_date_str = end_date_entry.get()
        timeframe_str=time_frame_var.get()

        try:
            # Convert the input date strings to datetime objects
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()

            # Create a DataCollector instance with the provided dates
            data_collector = DataCollector(start_date, end_date,timeframe_str,cont=False)

            # Start the data collection
            data_collector.start_collection()
            progress=data_collector.progress
           
        except ValueError:
            # Handle invalid date format
            messagebox.showerror("Invalid Date Format", "Please enter dates in the format YYYY-MM-DD.")

        data_collection_window.destroy()

    # Create the "Continue Collection" and "Collect New" buttons
    continue_collection_button = tk.Button(data_collection_window, text="Continue Collection", command=continue_collection_click)
    collect_new_button = tk.Button(data_collection_window, text="Collect New", command=collect_new_click)

    # Grid layout for the widgets
    start_date_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
    start_date_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
    end_date_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
    end_date_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
    time_frame_label.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
    time_frame_dropdown.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
    continue_collection_button.grid(row=3, column=1, padx=5, pady=10)
    collect_new_button.grid(row=3, column=0, padx=5, pady=10)
                
                


def train_model_gui(leverage,balance,equity):
    # Create the main window
    window.title("RL Training GUI")
    max_timesteps = tk.IntVar()

    # Function to handle the train button click event

    def train_button_click():
        
        ai_filter = ai_filter_var.get()
        if ai_filter == 'AI Trade Filter':
            ai_filter=True
        else:
            ai_filter=False
            
        env = tgym(leverage,balance,equity,ai_filter,training=True)
        env.reset()
        # Get the training option input
        train_mode = train_mode_var.get()
        max_timesteps = int(max_timesteps_var.get())

        progressbar = Progressbar(
            window, orient=tk.HORIZONTAL, length=200, mode='determinate')
        progressbar.grid(row=6, column=0, columnspan=2, pady=10)
        algorithm_name = algorithm_name_var.get()  # Get the selected algorithm name
        save_freq = save_freq_var.get()  # Get the selected save frequency
        model_directory = 'models/{}'.format(algorithm_name)

        if train_mode == "New Training":
            progressbar["value"] = 0
            window.update()
            train_model(ai_filter, algorithm_name, my_timesteps=1000, save_freq=save_freq,
                        model_directory=model_directory,
                        chckpcallback=my_callback(algorithm_name, save_freq), myoption="New Training", env=env,
                        progressbar=progressbar, max_timesteps=max_timesteps)
        elif train_mode == "Resume Training":
            progressbar["value"] = 0
            train_model(ai_filter, algorithm_name, my_timesteps=1000, save_freq=save_freq,
                        model_directory=model_directory,
                        chckpcallback=my_callback(algorithm_name, save_freq), myoption="Resume Training", env=env,
                        progressbar=progressbar, max_timesteps=max_timesteps)
        else:
            messagebox.showinfo(
                "Invalid Option", "Please enter a valid training (Resume or New Training).")

    # Create the checkbuttons
    max_timesteps_label = tk.Label(window, text="Maximum Steps:")
    max_timesteps_var = tk.Entry(window)

    # Create the algorithm_name dropdown menu and label
    train_mode_label = tk.Label(window, text="Training Mode:")
    train_modes = ['New Training', 'Resume Training']
    train_mode_var = tk.StringVar(
        value=train_modes[0])  # Set the initial value
    train_modes_dropdown = tk.OptionMenu(window, train_mode_var, *train_modes)
    train_modes_dropdown.config(width=15)

    # Create the algorithm_name dropdown menu and label
    ai_filter_label = tk.Label(window, text="Decision Filter:")
    ai_filter_modes = ['AI Trade Filter', 'Trade All']
    ai_filter_var = tk.StringVar(
        value=ai_filter_modes[0])  # Set the initial value
    ai_filter_dropdown = tk.OptionMenu(window, ai_filter_var, *ai_filter_modes)
    ai_filter_dropdown.config(width=15)

    # Create the algorithm_name dropdown menu and label
    algorithm_name_label = tk.Label(window, text="Algorithm Name:")
    algorithm_names = ['A2C', 'DQN', 'PPO', 'DDPG']
    algorithm_name_var = tk.StringVar(
        value=algorithm_names[0])  # Set the initial value
    algorithm_name_dropdown = tk.OptionMenu(
        window, algorithm_name_var, *algorithm_names)
    algorithm_name_dropdown.config(width=15)

    # Create the save_freq dropdown menu and label
    save_freq_label = tk.Label(window, text="Save Frequency:")
    save_freq_options = [1000, 10000, 100000]
    # Set the initial value
    save_freq_var = tk.IntVar(value=save_freq_options[0])
    save_freq_dropdown = tk.OptionMenu(
        window, save_freq_var, *save_freq_options)
    save_freq_dropdown.config(width=15)

    # Create the train button
    data_collection_button = tk.Button(window, text="Data Collect", command=data_collection_gui)
    train_button = tk.Button(window, text="Train", command=train_button_click)
    deploy_button = tk.Button(window, text="Deploy",
                              command=lambda: deploy_GUI(leverage,balance,equity,ai_filter_var.get(),algorithm_name_var.get()))

    # Grid layout for the widgets
    ai_filter_label.grid(row=1, column=0, sticky=tk.W)
    ai_filter_dropdown.grid(row=1, column=1)
    train_mode_label.grid(row=2, column=0, sticky=tk.W)
    train_modes_dropdown.grid(row=2, column=1)
    algorithm_name_label.grid(row=0, column=0, sticky=tk.W)
    algorithm_name_dropdown.grid(row=0, column=1)
    save_freq_label.grid(row=3, column=0, sticky=tk.W)
    save_freq_dropdown.grid(row=3, column=1)
    max_timesteps_label.grid(row=4, column=0, sticky=tk.W)
    max_timesteps_var.grid(row=4, column=1, stick=tk.W)
    # Updated grid layout for the train button
    train_button.grid(row=5, column=1, columnspan=2,
                      sticky=tk.W, pady=10, padx=45)
    deploy_button.grid(row=5, column=2, columnspan=2, pady=10,padx=10)
    data_collection_button.grid(row=5,column=0,columnspan=1,padx=10)

    # Start the GUI event loop
    window.mainloop()


if __name__ == '__main__':
    # Initializing the environment
    login_gui()


# changes to dashboard setting sizes
