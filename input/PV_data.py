import pandas as pd
import numpy as np
import plotly.express as px

def interpolate_pv_data(input_csv='PVdata_hourly.csv', output_csv='pv.csv'):
    # Read the CSV, parse 'time' as datetime
    df = pd.read_csv(input_csv, sep=';')
    df = df.drop(columns=['time'])
    #df.set_index('time', inplace=True)

    # Create a new minute-level time index for the whole year
    start_date = pd.Timestamp('2024-01-01 00:00')
    end_date = pd.Timestamp('2024-12-31 23:59')
    minute_index = pd.date_range(start=start_date, end=end_date, freq='min')
    df_minute = pd.DataFrame(index=minute_index, columns=df.columns)
    
    # Interpolate each column with noise
    for col in df.columns:
        # Reindex and interpolate
        data = np.array(df[col])
        data_min = np.array(365*24*60 * [0])  # Initialize with zeros for the whole year
#        print(data[0])
        d_prev = 0
        for d in range(365):
            ##print(d, col)
            for h in range(24):
                d_next = data[d * 24 + h + 1] if (d * 24 + h + 1) < len(data) else data[-1]
                for m in range(60):
                    current_delta = d_next - d_prev
                    current_time_delta = 1/(60-m-1) if m < 59 else 1
                    d_cur = d_prev + current_delta * current_time_delta
                    if m == 0: d_cur = data[d * 24 + h]
                    else:
                        noise = np.random.normal(0, 0.05 * abs(current_delta))
                        d_cur += noise
                        if d_cur < 0: d_cur = 0
                        if d_cur > 1000: d_cur = 1000  

                    #time = pd.Timestamp(f'2021-{month}-{day} {h}:{m:02d}')
                    #df_minute.iloc[d*24*60+h*60+m].loc[col] = d_cur
                    data_min[d * 24 * 60 + h * 60 + m] = d_cur
                    d_prev = d_cur 
        data_min = np.concatenate([data_min, np.zeros(1440)])
        df_minute[col] = data_min
    # Filter for July (2021-07-01 to 2021-07-31)
    july_mask = (df_minute.index >= '2024-07-01') & (df_minute.index < '2024-08-01')
    df_july_minute = df_minute.loc[july_mask]

    # Prepare initial data for July (assuming hourly data, with a 'time' column)
    july_initial_mask = (df.index >= 24*181) & (df.index < 24*212)
    df_july_initial = df.loc[july_initial_mask]
    # Reconstruct x-axis (timestamps) for hourly data in July
    july_hours = pd.date_range(start='2024-07-01 00:00', end='2024-07-31 23:00', freq='h')
    # Plot both interpolated and initial data for July
    import plotly.graph_objects as go

    fig = go.Figure()
    color_map = px.colors.qualitative.Plotly  # Use Plotly's qualitative color set
    for i, col in enumerate(df_july_minute.columns):
        color = color_map[i % len(color_map)]
        fig.add_trace(go.Scatter(
            x=df_july_minute.index, y=df_july_minute[col],
            mode='lines', name=f'{col} (minute)',
            line=dict(color=color)
        ))
        fig.add_trace(go.Scatter(
            x=july_hours,
            y=df_july_initial[col],
            mode='markers', name=f'{col} (hourly)',
            marker=dict(color=color, size=10)  # Increased marker size
        ))
    fig.update_layout(title='PV Data for July: Interpolated (minute) and Initial (hourly)')
    fig.show()

    # Matplotlib plots for July 20 and July 15 using indices, not timestamps
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for plot_idx, day_num in enumerate([15, 20]):
        # Calculate start and end indices for minute data (July starts at index 0)
        minute_start_idx = (day_num - 1) * 24 * 60
        minute_end_idx = minute_start_idx + 24 * 60
        df_day_minute = df_july_minute.iloc[minute_start_idx:minute_end_idx]

        # Calculate start and end indices for hourly data (July starts at index 0)
        hour_start_idx = (day_num - 1) * 24
        hour_end_idx = hour_start_idx + 24
        df_day_hourly = df_july_initial.iloc[hour_start_idx:hour_end_idx]

        ax = axes[plot_idx]
        x_minute = np.arange(24 * 60) / 60
        x_hour = np.arange(24)
        for i, col in enumerate(df_day_minute.columns):
            color = color_map[i % len(color_map)]
            ax.plot(x_minute, df_day_minute[col].values, label=f'{col} (minute)', color=color)
            ax.scatter(x_hour, df_day_hourly[col].values, label=f'{col} (hourly)', color=color, s=40, marker='o')
        ax.set_title(f'July {day_num}')
        ax.set_xlabel('Hour of Day')
        ax.set_xlim(0, 24)
        ax.set_ylim(0, 800)
        ax.set_xticks(np.arange(0, 25, 2))
        if plot_idx == 0:
            ax.set_ylabel('PV Value')
        ax.legend()
    plt.tight_layout()
    plt.show()
    # Save the interpolated minute-level data for the whole year to CSV
    df_minute.to_csv(output_csv)

# Example usage:
interpolate_pv_data()