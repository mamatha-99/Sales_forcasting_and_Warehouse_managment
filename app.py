#######################
# Import libraries
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
from nixtla import NixtlaClient

#######################
# Page configuration
st.set_page_config(
    page_title="Sales forcasting",
    page_icon="📈 ",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")

#######################
# CSS styling
st.markdown("""
<style>

[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}

[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
}

[data-testid="stMetric"] {
    background-color: #393939;
    text-align: center;
    padding: 15px 0;
}

[data-testid="stMetricLabel"] {
  display: flex;
  justify-content: center;
  align-items: center;
}

[data-testid="stMetricDeltaIcon-Up"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}

</style>
""", unsafe_allow_html=True)


#######################
# Load data
df1 = pd.read_csv('combined_df.csv')
df1['Date'] = pd.to_datetime(df1['Date'])


# Model key and initiation
nixtla_client = NixtlaClient(
    # defaults to os.environ.get("NIXTLA_API_KEY")
    api_key = 'nixak-odp8YhH8CEjt2wGiZ5zP5XCHoeMAFCj0hGGzpBDIA2QIcEZz4RqI98kxn7dFuRvbAIQaX6OXfDmL7vFL'
)

#######################
# Sidebar
with st.sidebar:
    st.title('📈  Sales Forecasting and Optimizing Warehouse Operations for Walmart Stores')
    
    
    store_list = list(df1['Store'].unique())
    store_list.insert(0, 'All')
    default_year = store_list[0]
    
    selected_store = st.selectbox('Select store', store_list, index=store_list.index(default_year))
    department_list = list(df1.query('Store == @selected_store')['Dept'].unique())
    department_list.insert(0, '-')
    default_department = department_list[0]
    selected_department = st.selectbox('Select department', department_list, index=department_list.index(default_department), disabled=(selected_store == 'All'))
    

#######################
# Plots

# Line plot for all stores
def line_plot(data):
    
    df = data.copy()

    df = df.groupby(['Date'])['Weekly_Sales'].mean().reset_index()

    fig = px.line(df, x='Date', y='Weekly_Sales', title='Overal Sales of 45 stores')

    # figure_size
    fig.update_layout(
        width=1000,
        height=250,
    )

    return fig

# line plot for storewise
def line_plot_store(data, store_number):
    
      df = data.copy()

      df= df.query('Store == @store_number')


      df = df.groupby(['Date'])['Weekly_Sales'].mean().reset_index()

      fig = px.line(df, x='Date', y='Weekly_Sales', title= f'Sales Over Time for Store {store_number}')

      # figure_size
      fig.update_layout(
          width=1000,
          height=250,
      )

      return fig

# line plot for dept

def line_plot_dept(data, store_number, dept_number):
      df = data.copy()

      df= df.query('Store == @store_number and Dept == @dept_number')
      
      
      fig = px.line(df, x='Date', y='Weekly_Sales', title= f'Sales Over Time for Department {dept_number} in  {store_number} store.')

      # figure_size
      fig.update_layout(
          width=1000,
          height=250,
      )


      return fig

# Overall forcaste 
def get_forecasts(data):

      df= data.copy()

      input_df = df.groupby(['Date'])['Weekly_Sales'].mean().reset_index()

      fcst_store = nixtla_client.forecast(
          df= input_df,
          h = 24,                                  #  Forecast horizon of 24 timestamps 24 weeks 6 months
          freq='W-FRI',                           # Frequency of date is Weekly
          time_col='Date',
          model='timegpt-1-long-horizon',         # Long horizon model
          target_col='Weekly_Sales',
          finetune_steps=15,                       # Model will go through 5 iterations of training
          level = [50,90,80],
          add_history = True
      )

      fig = nixtla_client.plot(
          df = input_df,
          forecasts_df = fcst_store,
          time_col = 'Date',
          target_col = 'Weekly_Sales',
          engine = 'plotly',
          #max_insample_length= 24,
          level = [50,90,80]
        )

      fig.update_layout(
            title_text="Sales forcast for next 6 months",  # Your existing title
            height=250, 
            width=1000 
        )
      return fig
  
# Store wise forcasting
def get_forecasts_Store(data, store_number):
    df = data.copy()

    df= data.groupby(['Date','Store'])['Weekly_Sales'].mean().reset_index()

    input_df = df.query('Store == 33')

    fcst_store = nixtla_client.forecast(
    df= input_df,
    h = 24,                                  #  Forecast horizon of 24 timestamps 24 weeks 6 months
    freq='W-FRI',                           # Frequency of date is Weekly
    time_col='Date',
    model='timegpt-1-long-horizon',         # Long horizon model
    target_col='Weekly_Sales',
    finetune_steps=15,                       # Model will go through 5 iterations of training
    level = [50,90,80],
    add_history = True
    )

    fig = nixtla_client.plot(
        df = input_df,
        forecasts_df = fcst_store,
        time_col = 'Date',
        target_col = 'Weekly_Sales',
        engine = 'plotly',
        #max_insample_length= 24,
        level = [50,90,80]
      )

    fig.update_layout(
          title_text=f"{store_number} Store's sales forcast for next 6 months ",  # Your existing title
          height=250, 
          width=1000 
      )
    return fig

# Store --> dept wise

def get_forecasts_Dept(data, store_number, dept_number):
    df= data.copy() # Create a copy of combined_df instead of grouping directly


    df = df.query('Store == @store_number') # Now you can use .query() on the DataFrame and select store number

    df['Dept'].nunique()  # number of departments a store has

    input_df = df.query('Dept == @dept_number') # select dep_num


    try :

      fcst_store = nixtla_client.forecast(
          df= input_df,
          h = 24,                                  #  Forecast horizon of 24 timestamps 24 weeks 6 months
          freq='W-FRI',                           # Frequency of date is Weekly
          time_col='Date',
          model='timegpt-1-long-horizon',         # Long horizon model
          target_col='Weekly_Sales',
          finetune_steps=15,                       # Model will go through 5 iterations of training
          level = [50,90,80],
          #add_history = True
      )

    except ValueError as e :
      print(f"Deparment number {dept_number} is not on sales for entier time to forecaste")

    fig = nixtla_client.plot(
        df = input_df,
        forecasts_df = fcst_store,
        time_col = 'Date',
        target_col = 'Weekly_Sales',
        engine = 'plotly',
        #max_insample_length= 24,
        level = [50,90,80]
      )

    fig.update_layout(
          title_text=f" Next 6 months forecast for Department {dept_number} in Store {store_number} ",  # Your existing title
          height=250, 
          width=1000 
      )

    return fig



    




# Calculate Sales Velocity
def calculate_sales_velocity(df, store_number=None, dept_number=None):
    """
    Calculates and normalizes sales velocity for overall data, store, or department.

    Args:
    df (pd.DataFrame): The input DataFrame.
    store_number (int, optional): The store number to filter by. Defaults to None.
    dept_number (int, optional): The department number to filter by. Defaults to None.

    Returns:
    float: The normalized sales velocity.
    """
    velocity_df = df.copy()
    
    if store_number and dept_number:
        velocity_df = velocity_df.query('Store == @store_number and Dept == @dept_number')
    elif store_number:
        velocity_df = velocity_df.query('Store == @store_number')
    
    velocity_df = velocity_df.groupby(['Date'])['Weekly_Sales'].mean().reset_index()
    
    n_days = velocity_df["Date"].nunique()
    Sales_n = velocity_df["Weekly_Sales"].sum()
    win_rate = 1  # Assuming since every week has sales
    
    sales_velocity = (Sales_n * win_rate) / n_days
    
    norm = (sales_velocity - velocity_df["Weekly_Sales"].min()) / (velocity_df["Weekly_Sales"].max() - velocity_df["Weekly_Sales"].min())

     # Format norm to two decimal places
    formatted_norm = f"{norm:.2f}"
    
    return formatted_norm

# Calculate highest sales month
def get_highest_sales_month(df, store_number=None, dept_number=None):
    """
    Gets the month with the highest average weekly sales for overall data, store, or department.

    Args:
        df (pd.DataFrame): The input DataFrame.
        store_number (int, optional): The store number to filter by. Defaults to None.
        dept_number (int, optional): The department number to filter by. Defaults to None.

    Returns:
        str: The month with the highest average weekly sales (e.g., "DEC").
    """
    df1 = df.copy()

    if store_number and dept_number:
        df1 = df1.query('Store == @store_number and Dept == @dept_number')
    elif store_number:
        df1 = df1.query('Store == @store_number')
        df1 = df1.groupby(['Date', 'Store'])['Weekly_Sales'].mean().reset_index()
    else:
        df1 = df1.groupby(['Date'])['Weekly_Sales'].mean().reset_index()

    df1['Month'] = df1['Date'].dt.strftime('%b').str.upper()
    df1 = df1.groupby(['Month'])['Weekly_Sales'].mean().reset_index()
    res = df1.sort_values(by='Weekly_Sales', ascending=False).head(1)

    return res['Month'].values[0]

# Top 10 Stores by Sales
def get_top_10_stores(df):
    df = df.groupby(['Store'])['Weekly_Sales'].mean().reset_index()
    df = df.sort_values(by='Weekly_Sales', ascending=False).head(10)
    df['Weekly_Sales'] = df['Weekly_Sales'].map('{:.2f}'.format)  # Apply formatting
    return df

# Top 10 Departments by Sales
def get_top_10_departments(df, store_number=None):
    if store_number:
        df = df.query('Store == @store_number')
    df = df.groupby(['Dept'])['Weekly_Sales'].mean().reset_index()
    df = df.sort_values(by='Weekly_Sales', ascending=False).head(10)
    df['Weekly_Sales'] = df['Weekly_Sales'].map('{:.2f}'.format)  # Apply formatting
    return df


#######################
# Dashboard Main Panel
col = st.columns((1.5, 4.5, 2), gap='medium')

with col[0]:
    st.markdown('#### Sales Velocity')
    if selected_store == 'All':
        sales_velocity = calculate_sales_velocity(df1)
    elif selected_department == '-':
        sales_velocity = calculate_sales_velocity(df1, selected_store)
    else:
        sales_velocity = calculate_sales_velocity(df1, selected_store, selected_department)
    
    st.metric(label="Sales Velocity", value=sales_velocity)

    
    st.markdown('#### Highest Sales Month')
    if selected_store == 'All':
        highest_sales_month = get_highest_sales_month(df1)
    elif selected_department == '-':
        highest_sales_month = get_highest_sales_month(df1, selected_store)
    else:
        highest_sales_month = get_highest_sales_month(df1, selected_store, selected_department)
        
    st.metric(label="Highest Sales Month", value=highest_sales_month)


# Add the Forecast button at the bottom
if st.button('Forecast') != True:
    with col[1]:
        
            if selected_store == 'All':
                st.markdown('#### Store Sales')
                st.plotly_chart(line_plot(df1))
            elif selected_department == '-':
                st.markdown('#### Store Sales')
                st.plotly_chart(line_plot_store(df1,selected_store))
            else :
                st.markdown('#### Store Sales')
                st.plotly_chart(line_plot_store(df1,selected_store))
                st.markdown('#### Department Wise Sales')
                st.plotly_chart(line_plot_dept(df1,selected_store,selected_department))
else:
    with col[1]:
         if selected_store == 'All':
                st.markdown('#### Store Sales')
                st.plotly_chart(get_forecasts(df1))
         elif selected_department == '-':
                st.markdown('#### Store Sales')
                st.plotly_chart(get_forecasts_Store(df1,selected_store))
         else :
                st.markdown('#### Store Sales')
                st.plotly_chart(get_forecasts_Store(df1,selected_store))
                st.markdown('#### Department Wise Sales')
                st.plotly_chart(get_forecasts_Dept(df1,selected_store,selected_department))
        
with col[2]:
    
    if selected_store == 'All':
        st.markdown('#### Top 10 Stores by Sales')
        top_stores = get_top_10_stores(df1)
        st.dataframe(top_stores,
                     column_order=("Store", "Weekly_Sales"),
                     hide_index=True,
                     width= None,
                     column_config={
                        "Store": st.column_config.TextColumn("Store"),
                        "Weekly_Sales": st.column_config.TextColumn("Weekly Sales")
                        }
                     )
    else:
        st.markdown('#### Top 10 Departments by Sales')
        top_departments = get_top_10_departments(df1, selected_store)
        st.dataframe(top_departments,
                     column_order=("Dept", "Weekly_Sales"),
                     hide_index=True,
                     width=None,
                     column_config={
                        "Dept": st.column_config.TextColumn("Department"),
                        "Weekly_Sales": st.column_config.TextColumn("Weekly Sales")
                        }
                     )
        
