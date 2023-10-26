import streamlit as st
from streamlit_option_menu import option_menu
import requests
from streamlit_lottie import st_lottie
import streamlit as st
import time
from streamlit_option_menu import option_menu
import plotly.express as px
import requests
import pandas as pd
import numpy as np
from streamlit_lottie import st_lottie
import pandas as pd
import numpy as np
import yfinance as yf
from pykeen.pipeline import pipeline
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from datetime import date
from prophet.plot import plot_plotly
from textblob import TextBlob

#=================================================================================================================

#data and data preprosessing
data = pd.read_csv('All India Consumer Price Index.csv')
data = data.dropna()
data.reset_index(drop=True, inplace=True)

#=================================================================================================================
st.set_page_config(layout="wide")

# Load Lottie animation
# Define a function to load Lottie animation
@st.cache_data
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


logo = load_lottieurl("https://lottie.host/8dfc14bb-34ec-493d-b169-64e8e93d9280/e66bPWSgiA.json")
empty=load_lottieurl("https://assets8.lottiefiles.com/datafiles/vhvOcuUkH41HdrL/data.json")
care=load_lottieurl("https://lottie.host/406116d5-df45-4be3-9996-8eb9be23ab42/Shkh7baebx.json")
predict1=load_lottieurl("https://lottie.host/bf98a5f9-d2c6-4b45-be7a-e76fede3cb03/5zUnocAWuz.json")
predict2=load_lottieurl("https://lottie.host/1bd53aef-a7da-49f9-b40e-ab51d6d2c649/rQbboVJeCm.json")
predict3=load_lottieurl("https://lottie.host/4166f5a0-6027-4a6e-a9d2-8b821ea7953c/4YskF6AYNK.json")
home_one=load_lottieurl("https://lottie.host/49835fbf-b640-42fb-b578-26e01c2830e4/r9aUmHIr9E.json")
home_two=load_lottieurl("https://lottie.host/e411b9e4-f435-4f8c-8320-e04563e89f4f/TxAvqHQLnR.json")
home_three=load_lottieurl("https://lottie.host/30896b27-434e-432a-9cbf-3d2d63f3a50e/EG2QYZ4vwt.json")
stock1=load_lottieurl("https://lottie.host/06610354-3053-4163-98bc-f09291da50c7/ecPmg6avJ9.json")
stock2=load_lottieurl("https://lottie.host/18785d74-ed1c-40d1-8dbd-f8ae6c95ce3a/VRGPSNvFRR.json")
stock3=load_lottieurl("https://lottie.host/1ec775f6-1812-4fee-9283-796cf36b022e/jx6bAgzxUD.json")
stock4=load_lottieurl("https://lottie.host/342fd00f-ab5d-48b7-8e8f-39b9a13f9d14/YgfzRCVL6F.json")


#=================================================================================================================
# Define functions for different pages


#=================================================================================================================
def page_1():
    with st.container():
        st.write("___")
        image_column, text_column = st.columns((1, 2))
        with text_column:
                st.write("""
                <div style="text-align: justify; font-size: 35px;">
                <p><b><h3>Understanding Investment</h3></b></p>

                <ul>
                    <li><b>What is Investment?</b></li>
                </ul>

                <p>Investment is the process of allocating your financial resources into various assets or ventures with the 
                expectation of achieving future financial returns. It plays a crucial role in building wealth and securing 
                financial stability for individuals and organizations.</p>

                <ul>
                    <li><b>Importance of Investment</b></li>
                </ul>

                <p>Investing is essential for several reasons:</p>

                <ul>
                    <li><b>Wealth Growth:</b> Properly managed investments can grow your wealth over time through capital 
                    appreciation and income generation.</li>
                    <li><b>Financial Security:</b> Investments can provide a safety net for unexpected expenses and future 
                    financial needs, such as retirement or education.</li>
                    <li><b>Beat Inflation:</b> Investing helps your money outpace inflation, preserving your purchasing power 
                    and ensuring your money doesn't lose value over time.</li>
                    <li><b>Financial Goals:</b> Investments enable you to work toward specific financial goals, such as buying a 
                    home, funding a business, or traveling the world.</li>
                </ul>

                <ul>
                    <li><b>Types of Investments</b></li>
                </ul>

                <p>Investments come in various forms, including:</p>

                <ul>
                    <li><b>Stocks:</b> Ownership shares in a company that can potentially appreciate in value and provide dividends.</li>
                    <li><b>Bonds:</b> Debt securities that pay periodic interest and return the principal at maturity.</li>
                    <li><b>Real Estate:</b> Investment in properties for rental income and potential capital appreciation.</li>
                    <li><b>Business Ownership:</b> Owning and operating a business or investing in an existing one.</li>
                </ul>

                <ul>
                    <li><b>Investment Considerations</b></li>
                </ul>

                <p>When investing, it's essential to consider factors like risk tolerance, time horizon, and diversification to 
                create a well-balanced investment portfolio.</p>

                <p>Whether you're planning for the long-term or looking to achieve short-term financial objectives, 
                making informed investment decisions is a key part of your financial journey.</p>
                </div>
                """, unsafe_allow_html=True)
        with image_column:
            st_lottie(home_one, height=500, key="under_cos")

        with st.container():
            st.write("___")
            image_column2, text_column2 = st.columns((2, 1))
            with text_column2:
                st.write("##")
                st.write("##")
                st.write("##")
                st.write("##")
                st.write("##")
                st.write("##")
                st_lottie(home_two, height=500, key="under_coos")
            with image_column2:
                st.write("""
                    <div style="text-align: justify; font-size: 35px;">
                    <p><b><h3>Understanding CPI</h3></b></p>

                    <p>The Consumer Price Index (CPI) is a key economic indicator that measures the average change over 
                    time in the prices paid by urban consumers for a market basket of consumer goods and services. The 
                    CPI is used to monitor inflation and assess the cost of living for the general population.</p>

                    <p>The CPI serves as an essential tool for economists, policymakers, and individuals to make informed 
                    decisions related to economic planning, wage adjustments, and investment strategies.</p>

                    <p><b><h3>Calculation of CPI:</h3></b></p>

                    <p>The CPI is calculated by comparing the prices of a fixed basket of goods and services over time. The 
                    basket includes various items, such as food, housing, transportation, healthcare, and education, 
                    representing the typical spending patterns of urban consumers.</p>

                    <p>The CPI value is normalized to a base year, usually set to 100, and subsequent values reflect changes 
                    in prices relative to the base year.</p>

                    <p><b><h3>Significance of CPI:</h3></b></p>

                    <p>The CPI has several important implications:</p>

                    <ul>
                        <li><b>Inflation Monitoring:</b> Rising CPI indicates inflationary pressures in the economy, which can 
                        erode the purchasing power of money.</li>
                        <li><b>Cost of Living Adjustment:</b> CPI data is used to determine cost-of-living adjustments for 
                        salaries, wages, and government benefits, ensuring they keep pace with changes in prices.</li>
                        <li><b>Economic Policy Decisions:</b> Policymakers use CPI data to formulate economic policies, such as 
                        adjusting interest rates and taxation, to stabilize the economy.</li>
                        <li><b>Economic Analysis:</b> Economists use CPI trends to analyze economic performance, forecast future 
                        price movements, and understand consumer behavior.</li>
                    </ul>

                    <p><b><h3>CPI as a Measure of Inflation:</h3></b></p>

                    <p>The CPI serves as a primary measure of inflation in an economy. A higher CPI value indicates 
                    rising prices, while a lower value suggests deflation or falling prices.</p>

                    <p><b><h3>Inflationary Impact:</h3></b></p>

                    <p>Inflation impacts consumers, businesses, and the overall economy in several ways:</p>

                    <ul>
                        <li><b>Reduced Purchasing Power:</b> Rising prices reduce the real value of money, leading to decreased 
                        purchasing power for consumers.</li>
                        <li><b>Interest Rates:</b> Central banks use CPI data to set interest rates. High inflation may lead to 
                        higher interest rates to control spending and stabilize prices.</li>
                        <li><b>Investment Decisions:</b> Investors consider inflation when making investment decisions to 
                        preserve their wealth and achieve real returns.</li>
                    </ul>

                    <p><b><h3>CPI and Wage Growth:</h3></b></p>

                    <p>As the CPI increases, wages and salaries often need to be adjusted to maintain the standard of 
                    living for workers.</p>

                    <p>In the next sections, we will explore how the CPI evolves over different time periods and its 
                    implications for individuals and the economy.</p>
                    </div>
                    """, unsafe_allow_html=True)

        with st.container():
            st.write("___")
            image_column3, text_column3 = st.columns((1, 2))
            with text_column3:
                st.write("""
                        <div style="text-align: justify; font-size: 35px;">
                        <p><b><h3>Understanding Bank Fixed Deposits (FDs)</h3></b></p>

                        <ul>
                            <li><b>What are Bank Fixed Deposits?</b></li>
                        </ul>

                        <p>Bank Fixed Deposits, commonly known as FDs, are a type of investment offered by banks. When you invest in 
                        an FD, you deposit a lump sum amount with the bank for a specified period, known as the tenure. In return, 
                        the bank pays you a fixed interest rate, which is higher than regular savings accounts, for the duration 
                        of the FD.</p>

                        <ul>
                            <li><b>Benefits of Investing in Bank FDs</b></li>
                        </ul>

                        <p>Bank FDs offer several advantages:</p>

                        <ul>
                            <li><b>Capital Preservation:</b> FDs are considered a safe investment option as they are backed by the 
                            guarantee of the bank, making them low-risk.</li>
                            <li><b>Fixed Returns:</b> Investors receive a fixed interest rate, providing predictability and stability 
                            in income.</li>
                            <li><b>Flexible Tenures:</b> FDs offer flexibility in choosing the tenure, ranging from a few months to 
                            several years, allowing investors to align with their financial goals.</li>
                        </ul>

                        <p><b><h3>Understanding Stocks</h3></b></p>

                        <ul>
                            <li><b>What are Stocks?</b></li>
                        </ul>

                        <p>Stocks, also known as equities or shares, represent ownership in a company. When you buy stocks of a 
                        company, you become a shareholder and own a portion of that company. Stocks are traded on stock exchanges, 
                        and their prices can fluctuate based on various factors, including company performance, market sentiment, 
                        and economic conditions.</p>

                        <ul>
                            <li><b>Benefits of Investing in Stocks</b></li>
                        </ul>

                        <p>Investing in stocks offers several advantages:</p>

                        <ul>
                            <li><b>Potential for Capital Growth:</b> Stocks have the potential to appreciate in value over time, 
                            allowing investors to benefit from capital gains.</li>
                            <li><b>Dividend Income:</b> Many companies pay dividends to their shareholders, providing a source of 
                            regular income.</li>
                            <li><b>Ownership and Voting Rights:</b> Shareholders have a say in company decisions and can vote on 
                            important matters during shareholder meetings.</li>
                        </ul>

                        <p><b><h3>Understanding Investment</h3></b></p>

                        <ul>
                            <li><b>What is Investment?</b></li>
                        </ul>

                        <p>Investment is the process of allocating your financial resources into various assets or ventures with the 
                        expectation of achieving future financial returns. It plays a crucial role in building wealth and securing 
                        financial stability for individuals and organizations.</p>

                        <ul>
                            <li><b>Importance of Investment</b></li>
                        </ul>

                        <p>Investing is essential for several reasons:</p>

                        <ul>
                            <li><b>Wealth Growth:</b> Properly managed investments can grow your wealth over time through capital 
                            appreciation and income generation.</li>
                            <li><b>Financial Security:</b> Investments can provide a safety net for unexpected expenses and future 
                            financial needs, such as retirement or education.</li>
                            <li><b>Beat Inflation:</b> Investing helps your money outpace inflation, preserving your purchasing power 
                            and ensuring your money doesn't lose value over time.</li>
                            <li><b>Financial Goals:</b> Investments enable you to work toward specific financial goals, such as buying a 
                            home, funding a business, or traveling the world.</li>
                        </ul>

                        <p>These are some of the key investment options, including Bank FDs and stocks, and their advantages to consider 
                        in your financial planning.</p>
                        </div>
                        """, unsafe_allow_html=True)

            with image_column3:
                st.write("##")
                st.write("##")
                st.write("##")
                st.write("##")
                st.write("##")
                st.write("##")

                st_lottie(home_three, height=500, key="under_cosss")

def page_2():
    with st.container():
        st.write("___")
        image_column, text_column = st.columns((1, 2))
        with text_column:
            # Function to train the Prophet model for the selected category
            def train_prophet_model(category_data):
                model = Prophet()
                model.fit(category_data)
                return model


            # Function to make predictions for future dates
            def predict_future_cpi(model, future_date):
                future = pd.DataFrame({'ds': pd.to_datetime([future_date])})
                forecast = model.predict(future)
                return forecast['yhat'].iloc[0]


            # Function to calculate annualized return on investment
            def calculate_annualized_return(future_value, investment_amount, investment_date, withdrawal_date):
                time_period = (withdrawal_date - investment_date).days / 365
                if time_period <= 0:
                    return None
                annualized_return = (future_value / investment_amount) ** (1 / time_period) - 1
                return annualized_return


            # Streamlit App
            st.write("##")
            st.header("CPI Prediction and Investment Calculator (Inflation)")
            st.write("##")


            # Dropdown to select the category
            selected_category = st.selectbox("Select Category",
                                             data.columns[3:-1])  # Adjust column range based on your data

            # Filter data for the selected category
            category_data = data[['Year', 'Month', selected_category]]
            category_data.columns = ['ds', 'dummy', 'y']  # Rename columns as required by Prophet

            # Check if there is sufficient data for the selected category
            if len(category_data) == 0:
                st.error("No data available for the selected category.")
            else:
                # Display attractive plot using Plotly
                fig = px.line(data_frame=category_data, x='ds', y='y', labels={'ds': 'Date', 'y': 'CPI'})
                st.plotly_chart(fig)

                # Train Prophet model for the selected category
                model = train_prophet_model(category_data)

                # Date input for future prediction
                future_date = st.date_input("Select Future Date")

                # Predict CPI button
                if st.button("Predict CPI"):
                    # Make prediction for the future date
                    predicted_cpi = predict_future_cpi(model, future_date)
                    st.write(f"Predicted CPI for {selected_category} on {future_date}: {predicted_cpi:.2f}")
                    st.write("***")
                # Investment amount input (optional)
                investment_amount = st.number_input("Enter the Investment Amount (Optional)", min_value=0, value=1000)

                # Investment date input (optional)
                investment_date = st.date_input("Select Investment Date (Optional)")

                # Withdrawal date input (optional)
                withdrawal_date = st.date_input("Select Withdrawal Date (Optional)")

                # Calculate Future Value button
                if investment_amount > 0 and investment_date and withdrawal_date and st.button(
                        "Calculate Future Value"):
                    # Make prediction for the future date
                    predicted_cpi = predict_future_cpi(model, future_date)

                    # Calculate future value
                    future_value = investment_amount * (1 + predicted_cpi / 100) ** (
                            (withdrawal_date - investment_date).days / 465)

                    # Calculate annualized return
                    annualized_return = calculate_annualized_return(future_value, investment_amount, investment_date,
                                                                    withdrawal_date)
                    if annualized_return is not None:
                        st.success(f"Future Value of Investment: {future_value:.2f}")
                        st.success(f"Annualized Return on Investment: {annualized_return:.2%}")
                    else:
                        st.warning("Invalid investment duration. Please make sure the investment duration is positive.")
            st.write("Note: Maximum Values Take in Consideration")
        with image_column:
            st.write("##")
            st.write("##")
            st_lottie(predict2, height=500, key="under_cos")
            st.write("##")
            st.write("##")
            st.write("##")
            st.write("##")
            st.write("##")
            st_lottie(predict3, height=500, key="Predict3")

    with st.container():
        st.write("___")
        # Synthetic data for gold returns, mutual fund returns, and bank FD returns from 2010 to 2021
        years = np.arange(2010, 2022)

        # Replace with your actual data for each column
        gold_returns = [9.0, 10.0, 8.5, 9.5, 11.0, 10.5, 9.8, 8.9, 8.7, 9.2, 9.6, 10.1]
        mutual_fund_returns = [8.2, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 6.1, 6.3, 6.7, 7.0, 7.2]
        bank_fd_returns = [6.0, 5.8, 5.5, 5.4, 5.2, 5.1, 4.9, 4.8, 4.6, 4.5, 4.4, 4.3]

        data1 = pd.DataFrame({'Year': years, 'Gold Returns': gold_returns, 'Mutual Fund Returns': mutual_fund_returns,
                             'Bank FD Returns (Average)': bank_fd_returns})

        # Streamlit app
        st.header('Historical Investment Returns on Gold, Mutual Funds and FD')

        # Select columns to display
        selected_columns = st.multiselect('Select columns to display:', data1.columns)

        if selected_columns:
            fig = px.line(data1, x='Year', y=selected_columns, title='Investment Returns Over Time')
            fig.update_layout(width=1500, height=500)
            st.plotly_chart(fig)

#=================================================================================================================
def page_3():
        st.write("___")

        # Define the menu items and their corresponding functions
        menu_items = {
            "Data Overview": lambda: display_data(),
            "News": lambda: display_news(),
            "Stock Safety": lambda: display_Safety(),
            "Empty": lambda: display_empty(),
        }

        # Function to display the Home page content
        def display_data():
            st.header('Sector Wise Nifty 100 Stock Dashboard')

            # Use the @st.cache decorator to cache the function's output based on its input
            @st.cache_data
            def load_data(ticker):
                data = yf.download(ticker, start="2010-01-01", end=pd.Timestamp.today().strftime("%Y-%m-%d"))
                data.reset_index(inplace=True)
                return data

            def get_nifty_100_stocks():
                # List of Nifty 100 stocks with their respective sectors
                nifty100_stocks = {
                    'Technology': ['TCS.NS', 'INFY.NS', 'HCLTECH.NS', 'TECHM.NS', 'WIPRO.NS'],
                    'Oil': ['RELIANCE.NS', 'ONGC.NS', 'IOC.NS'],
                    'Finance': ['HDFCBANK.NS', 'ICICIBANK.NS', 'AXISBANK.NS', 'SBI.NS'],
                    'Banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'AXISBANK.NS', 'SBI.NS', 'KOTAKBANK.NS'],
                    'Automotive': ['TATAMOTORS.NS', 'MARUTI.NS', 'HEROMOTOCO.NS'],
                    'Consumer Durables': ['HUL.NS', 'ITC.NS', 'BAJAJFINSV.NS'],
                    'FMCG': ['HUL.NS', 'ITC.NS', 'BAJAJFINSV.NS'],
                    'Healthcare': ['SUNPHARMA.NS', 'DRREDDY.NS', 'INFY.NS'],
                    'Metals': ['TATASTEEL.NS', 'VEDL.NS', 'JSWSTEEL.NS'],
                    'Utilities': ['NTPC.NS', 'POWERGRID.NS', 'BHEL.NS'],
                    'IT': ['TCS.NS', 'INFY.NS', 'HCLTECH.NS', 'TECHM.NS', 'WIPRO.NS'],
                }
                return nifty100_stocks

            def get_sector_data(selected_sector):
                nifty100_stocks = get_nifty_100_stocks()
                sector_stocks = nifty100_stocks.get(selected_sector, [])
                sector_data = pd.DataFrame(columns=['Stock', 'Current Price', 'Market Cap'])
                total_market_cap = 0

                for stock in sector_stocks:
                    data = load_data(stock)  # The cached data will be retrieved from the cache if available
                    if data is not None and not data.empty:
                        current_price = data['Close'].iloc[-1]
                        stock_info = yf.Ticker(stock)
                        market_cap = stock_info.info.get('marketCap', 'N/A')
                        sector_data = sector_data.append(
                            {'Stock': stock, 'Current Price': current_price, 'Market Cap': market_cap},
                            ignore_index=True)
                        total_market_cap += market_cap

                sector_data['Market Cap'] = pd.to_numeric(sector_data['Market Cap'],
                                                          errors='coerce')  # Convert Market Cap to numeric

                return sector_data, total_market_cap

            def generate_graph():
                x = np.linspace(0, 10, 100)
                y = np.sin(x)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='sin(x)'))
                fig.update_layout(title="Sample Graph", xaxis_title="X-axis", yaxis_title="Y-axis")
                return fig

            def main():
                nifty100_stocks = get_nifty_100_stocks()
                sectors = list(nifty100_stocks.keys())

                # Dropdown to select the sector
                selected_sector = st.selectbox("Select Sector", sectors)

                with st.container():
                    st.write("___")
                    image_column11, text_column11 = st.columns((2, 2))
                    with text_column11:
                        st.write("Sector Wise Nifty 100 Stock Dashboard")

                        # Get stocks in the selected sector
                        sector_data, total_market_cap = get_sector_data(selected_sector)

                        # Display the stocks in the selected sector with their current prices and market caps
                        st.write("##")
                        st.write("##")
                        st.subheader(f"Nifty 100 Stocks in {selected_sector}")

                        # Custom CSS styling for the data table
                        st.markdown("""
                        <style>
                        .dataframe {
                            font-size: 14px;
                            text-align: center;
                        }
                        th {
                            background-color: #f2f2f2;
                            font-size: 16px;
                            font-weight: bold;
                        }
                        td {
                            background-color: #f9f9f9;
                        }
                        </style>
                        """, unsafe_allow_html=True)

                        # Display data frame for sector_data
                        st.dataframe(sector_data)

                        # Display top 5 stocks in the selected sector by high value
                        st.subheader(f"Top 5 Stocks in {selected_sector} by High Value")
                        top_stocks_by_high_value = sector_data.nlargest(5, 'Current Price')
                        st.dataframe(top_stocks_by_high_value)

                    with image_column11:
                        # Bar chart to visualize the top 5 stocks by market cap
                        st.subheader(f"Top 5 Stocks in {selected_sector} - Market Cap")
                        top_stocks_by_market_cap = sector_data.nlargest(5, 'Market Cap')
                        fig_bar = px.bar(top_stocks_by_market_cap, x='Stock', y='Market Cap',
                                         labels={'Market Cap': 'Market Cap (INR)'},
                                         title=f"Top 5 Stocks in {selected_sector} by Market Cap")
                        fig_bar.update_layout(xaxis_title="Stock", yaxis_title="Market Cap (INR)", xaxis_tickangle=-45)
                        st.plotly_chart(fig_bar, use_container_width=True)

                        # Pie chart to show the weightage of all stocks in the selected sector based on market cap
                        sector_weights = top_stocks_by_market_cap['Market Cap'] / total_market_cap
                        pie_chart = go.Pie(labels=top_stocks_by_market_cap['Stock'], values=sector_weights,
                                           textinfo='label+percent',
                                           title=f"Weightage of Stocks in {selected_sector} by Market Cap")
                        fig_pie = go.Figure(pie_chart)
                        fig_pie.update_traces(marker=dict(colors=px.colors.qualitative.Plotly))
                        fig_pie.update_layout(title_x=0.5)
                        st.plotly_chart(fig_pie, use_container_width=True)

            if __name__ == '__main__':
                main()

        # Function to display the About page content
        def display_news():
            # Set Streamlit theme

            # Replace with your actual News API key
            NEWS_API_KEY = "9c1b3791f31345e68cf0b61f9887e102"

            # Add more stock symbols here
            stock_symbols = [
                "TCS", "RELIANCE", "HDFCBANK", "INFY", "ICICIBANK",
                "HINDUNILVR", "BAJFINANCE", "KOTAKBANK", "ITC", "ASIANPAINT",
                "HDFC", "LT", "M&M", "MARUTI", "NTPC",
                "ONGC", "POWERGRID", "RELIANCE", "SBIN", "SUNPHARMA",
                "TATAMOTORS", "TATASTEEL", "TCS", "TECHM", "TITAN",
                "ULTRACEMCO", "UPL", "VEDL", "WIPRO", "ZEEL", "JIO Financial Services"
            ]

            def fetch_news(stock_symbol, num_articles=10):
                url = f"https://newsapi.org/v2/everything?q={stock_symbol}&apiKey={NEWS_API_KEY}&pageSize={num_articles}"
                response = requests.get(url)
                data = response.json()
                return data.get("articles", [])

            def analyze_sentiment(article):
                text = article["title"]
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity

                if polarity > 0:
                    return "positive"
                elif polarity < 0:
                    return "negative"
                else:
                    return "neutral"

            def main():
                st.header("Indian Stock Market News Analysis")

                selected_stock = st.selectbox("Select a stock symbol", stock_symbols)

                st.write(f"Selected Stock: {selected_stock}")

                news_data = fetch_news(selected_stock, num_articles=10)

                st.write("Recent News:")
                for article in news_data:
                    sentiment = analyze_sentiment(article)
                    notation = "Positive" if sentiment == "positive" else "Negative"
                    article_title = article['title']
                    article_url = article['url']

                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; background-color: {'#7bbd7b' if sentiment == 'positive' else '#e07777'}; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                            <div>{article_title} ({notation} news)</div>
                            <div><a href="{article_url}" target="_blank"><button style="background-color: #3498db; color: white; border: none; padding: 5px 10px; border-radius: 3px;">Read More</button></a></div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            if __name__ == "__main__":
                main()

        # Function to display the Services page content
        def display_Safety():
            def get_stock_info(ticker):
                """Gets stock information for the given ticker."""
                stock = yf.Ticker(ticker + ".BO")  # Append ".BO" for Indian stocks
                return stock.info

            def calculate_safety_score(stock_info):
                """Calculates a safety score for the stock based on the P/E ratio."""
                # Define the maximum P/E ratio considered safe
                max_safe_pe_ratio = 20

                # Calculate the safety score based on P/E ratio
                pe_ratio = stock_info['trailingPE']
                if pe_ratio <= max_safe_pe_ratio:
                    return 1.0  # Stock is safe to buy
                else:
                    return max_safe_pe_ratio / pe_ratio

            def calculate_safe_buying_price(current_price, safety_score, num_days):
                """Calculates a safe buying price based on current price, safety score, and time frame."""
                # Define the safety discount factor based on the time frame (e.g., 15 days)
                discount_factor = (1 - (safety_score * 0.1)) ** (num_days / 365)
                safe_buying_price = current_price * discount_factor
                return max(safe_buying_price, current_price)

            def main():
                st.header('Stock Safety and Buying Price Calculator')
                ticker = st.text_input('Enter a stock ticker (e.g., TCS, RELIANCE):')

                if ticker:
                    stock_info = get_stock_info(ticker)

                    safety_score = calculate_safety_score(stock_info)
                    num_days = st.number_input('Enter the time frame (in days):', min_value=1, value=15)
                    current_price = stock_info['regularMarketOpen']
                    safe_buying_price = calculate_safe_buying_price(current_price, safety_score, num_days)

                    st.write(f"**Stock Information for {ticker}**")
                    st.write(f"Name: {stock_info['longName']}")
                    st.write(f"Sector: {stock_info['sector']}")
                    st.write(f"Market Cap: {stock_info['marketCap']:,} INR")
                    st.write(f"Current Price: {current_price:.2f} INR")

                    st.write(f"**Safety Assessment**")
                    st.write(f"P/E Ratio: {stock_info['trailingPE']:.2f}")
                    if safety_score == 1.0:
                        st.success('🚀 The stock is considered safe for investment.')
                    else:
                        st.warning('⚠️ The stock may have a high P/E ratio, further evaluation is recommended.')

                    st.write(f"**Recommended Safe Buying Price**")
                    st.write(f"Safe Buying Price (within {num_days} days): {safe_buying_price:.2f} INR")

            if __name__ == "__main__":
                main()

        # Function to display the Contact page content
        def display_empty():
            st.write("___empty_____")

        # Create the Streamlit app
        def main():
            selected_page = st.radio("Select a Page", list(menu_items.keys()))

            # Call the function corresponding to the selected page
            menu_items[selected_page]()

        if __name__ == "__main__":
            main()


#=================================================================================================================
def page_4():
    image_column, text_column = st.columns((1, 2))
    with text_column:
        START = "2010-01-01"
        TODAY = date.today().strftime("%Y-%m-%d")

        st.header('Stock Price Forecasting 📈')


        # List of Nifty 100 stocks
        nifty100_stocks = [
            'TCS', 'RIL', 'ICICIBANK', 'BAJAJFINSV', 'HDFCBANK', 'INFY', 'HINDUNILVR', 'KOTAKBANK', 'SBIN',
            'BAJFINANCE',
            'AXISBANK', 'ITC', 'LT', 'INDUSINDBK', 'ASIANPAINT', 'TITAN', 'SUNPHARMA', 'TATAMOTORS', 'HCLTECH',
            'DRREDDY',
            'POWERGRID', 'MARUTI', 'NESTLEIND', 'ONGC', 'TECHM', 'BAJAJFINSV', 'ULTRACEMCO', 'JSWSTEEL', 'BPCL', 'NTPC',
            'HDFCLIFE', 'BRITANNIA', 'DIVISLAB', 'GAIL', 'ADANIPORTS', 'COALINDIA', 'SHREECEM', 'HINDALCO', 'TATASTEEL',
            'IOC', 'GRASIM', 'CIPLA', 'SBIN', 'ADANIPORTS', 'HINDPETRO', 'ASIANPAINT', 'TCS', 'ONGC', 'TATAMOTORS',
            'IOC', 'TITAN', 'HINDALCO', 'TECHM', 'SBIN', 'NTPC', 'BPCL', 'SUNPHARMA', 'HINDUNILVR', 'UPL', 'GRASIM',
            'BRITANNIA', 'CIPLA', 'DIVISLAB', 'GAIL', 'JSWSTEEL', 'COALINDIA', 'EICHERMOT', 'MARUTI', 'SHREECEM',
            'TATASTEEL', 'BAJAJFINSV', 'ULTRACEMCO', 'HINDALCO', 'TATASTEEL', 'IOC', 'JSWSTEEL', 'AXISBANK',
            'HDFCBANK', 'TITAN', 'SBIN', 'NTPC', 'ONGC', 'HDFC', 'SBIN', 'INFY','IOB'
        ]
        selected_stock = st.selectbox('Select dataset for prediction', nifty100_stocks)

        n_years = st.slider('Years of prediction:📆', 1, 15)
        period = n_years * 365


        @st.cache_data
        def load_data(ticker):
            data = yf.download(ticker + ".NS", START, TODAY)
            data.reset_index(inplace=True)
            return data


        data_load_state = st.text('Loading data...⏳')
        data = load_data(selected_stock)
        data_load_state.text('Loading data... done!✔️')

        st.subheader('Raw data')
        st.write(data.tail())


        # Plot raw data
        def plot_raw_data():
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
            fig.layout.update(title_text='Time Series data with Rangeslider📊', xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)


        plot_raw_data()

        # Predict forecast with Prophet.
        df_train = data[['Date', 'Close']]
        df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

        m = Prophet(daily_seasonality=True)
        m.fit(df_train)
        future = m.make_future_dataframe(periods=period)
        forecast = m.predict(future)

        # Show and plot forecast
        st.subheader('Forecast data📏')
        st.write(forecast.tail())

        st.write(f'Forecast plot for {n_years} years 📅 ')
        fig1 = plot_plotly(m, forecast, xlabel="Date", ylabel="Close")
        st.plotly_chart(fig1)

    with image_column:
        st.write("##")
        st.write("##")
        st.write("##")
        st.write("##")
        st_lottie(stock1, height=500, key="under_cos")
        st.write("##")
        st.write("##")
        st.write("##")
        st.write("##")
        st.write("##")
        st.write("##")
        st.write("##")
        st.write("##")
        st.write("##")
        st.write("##")
        st.write("##")
        st.write("##")
        st.write("##")
        st.write("##")
        st.write("##")
        st.write("##")
        st.write("##")
        st.write("##")
        st.write("##")
        st.write("##")
        st.write("##")
        st.write("##")
        st.write("##")
        st.write("##")
        st.write("##")
        st.write("##")
        st.markdown("###")
        st.markdown("###")

        st_lottie(stock2, height=300, key="stock2")


    with st.container():
        st.write("___")
        image_column0, text_column0 = st.columns((2, 2))

        with image_column0:
            st.subheader("Forecast components")
            fig2 = m.plot_components(forecast)
            st.write(fig2)

            st.markdown("***")
            st.subheader("Analyse the drop in trend in March 2020 due to Lockdown")
            option = st.selectbox('Select the stock you want to analyse', nifty100_stocks)
            get_data = load_data(option)

            # Setting the range of base plot
            fig = px.line(get_data, x='Date', y='High',
                          title=f"{option}: Day's High Price during Phase 1 Lockdown(RED)(25 March – 14 April) and Phase 2 Lockdown (GREEN)(15 April – 3 May💯)",
                          range_x=['2020-01-01', '2020-06-30'])

            # Adding the shape in the dates
            fig.update_layout(
                shapes=[
                    # First phase Lockdown
                    dict(type="rect", xref="x", yref="paper", x0="2020-03-25", y0=0, x1="2020-04-14", y1=1,
                         fillcolor="Red",
                         opacity=0.5, layer="below", line_width=0, ),
                    # Second phase Lockdown
                    dict(type="rect", xref="x", yref="paper", x0="2020-04-15", y0=0, x1="2020-05-03", y1=1,
                         fillcolor="Green",
                         opacity=0.5, layer="below", line_width=0, )
                ])
            st.plotly_chart(fig)

            st.markdown("***")

        with text_column0:
            st.write("##")
            st.write("##")
            st.write("##")
            st.write("##")
            st_lottie(stock3, height=500, key="stock3")
            st.write("##")
            st.write("##")
            st.write("##")
            st.write("##")
            st.write("##")
            st.write("##")
            st.write("##")
            st.write("##")
            st.write("##")
            st.write("##")
            st.write("##")
            st_lottie(stock4, height=500, key="stock4")

#================================================================================================================

def page_5():
    st.title("PAGE 5")
    st.write("You selected PAGE 5 content.")
#=================================================================================================================
def page_6():
    st.title("PAGE 6")
    st.write("You selected PAGE 6 content.")


#=================================================================================================================
# Main Streamlit app
def main():

    image_column, text_column = st.columns((0.7, 2.3))
    with image_column:
        st_lottie(logo, height=240, key="under_log")
    with text_column:
        st.markdown("""
            <style>
                @font-face {
                    font-family: "a Astro Space";
                    src: url("https://fonts.cdnfonts.com/css/a-astro-space.font") format("woff2");
                    font-weight: normal;
                    font-style: normal;
                }
                h1 {
                    font-family: "a Astro Space";
                    font-size: 40px;
                    color: #000000;
                    text-decoration: underline;
                }
            </style>
        """, unsafe_allow_html=True)

        st.markdown('<h1>InvestAI: Connecting with Future Gains</h1>', unsafe_allow_html=True)
        st.write("##")


        # Display option menu and select the page to show
        selected_page = option_menu(
            menu_title=None,
            options=["PAGE 1", "PAGE 2", "PAGE 3", "PAGE 4", "PAGE 5", "PAGE 6"],
            icons=["book-half", "book-half", "book-half", "book-half", "book-half", "book-half"],
            orientation="horizontal",
            menu_icon="cast",
            default_index=0,
        )

    # Call the corresponding function based on the selected page
    if selected_page == "PAGE 1":
        page_1()
    elif selected_page == "PAGE 2":
        page_2()
    elif selected_page == "PAGE 3":
        page_3()
    elif selected_page == "PAGE 4":
        page_4()
    elif selected_page == "PAGE 5":
        page_5()
    elif selected_page == "PAGE 6":
        page_6()

    hide_streamlit_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
