import os
from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from werkzeug.utils import secure_filename
from sklearn.preprocessing import LabelEncoder
import mysql.connector
import numpy as np

pd.set_option('display.max_columns', None)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')

# Database configuration
db_config = {
    'user': 'root',
    'password': '123456789kobi.',
    'host': '127.0.0.1',
    'database': 'sample'
}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def get_db_connection():
    conn = mysql.connector.connect(**db_config)
    return conn

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'files[]' not in request.files:
        return redirect(request.url)

    files = request.files.getlist('files[]')
    platform_filter = request.form.get('platform_filter')  # Get the platform filter from the form

    plot_urls = []
    summary_stats = {}
    insights = []
    dataframes_html = []
    summary_stats_html = []

    for file in files:
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            try:
                file_plot_urls, file_summary_stats, file_insights, dataframe_html = create_insights(filepath, platform_filter)
                plot_urls.extend(file_plot_urls)
                summary_stats[filename] = file_summary_stats
                insights.extend(file_insights)
                dataframes_html.append({'filename': filename, 'table': dataframe_html})
                save_insights_to_db(filename, file_insights)
                summary_stats_html.append({'filename': filename, 'summary': file_summary_stats})
            except Exception as e:
                return render_template('index.html', error=str(e))

    summary_plot_url = generate_summary_plot(summary_stats)
    insights_plot_url = generate_insights_plot(insights)

    return render_template(
        'index.html', 
        plot_urls=plot_urls, 
        summary_plot_url=summary_plot_url, 
        insights_plot_url=insights_plot_url, 
        dataframes_html=dataframes_html,
        summary_stats_html=summary_stats_html
    )

def create_insights(filepath, platform_filter=None):
    data = pd.read_csv(filepath)
    data['Year'] = data['Year'].astype('Int64')  # Convert 'Year' to integer type

    if platform_filter:
        data = data[data['Platform'] == platform_filter]

    # Convert all NumPy types to native Python types
    data = data.where(pd.notnull(data), None)  # Replace NaNs with None
    data = data.applymap(lambda x: x.item() if isinstance(x, np.generic) else x)
    
    dataframe_html = data.head(5).to_html(classes='table table-striped', index=False)  # Limit to first 5 rows

    drop_row_index = data[data['Year'] > 2015].index
    data = data.drop(drop_row_index)

    for column in data.select_dtypes(include=['object']).columns:
        try:
            data[column] = pd.to_numeric(data[column])
        except ValueError:
            data[column] = LabelEncoder().fit_transform(data[column])

    numeric_data = data.select_dtypes(include='number')

    if numeric_data.empty:
        raise ValueError("No numeric data found in the CSV file.")

    plot_urls = []
    summary_stats = numeric_data.describe().transpose().to_dict()
    insights = []

    if 'Year' in data.columns and 'Genre' in data.columns:
        plt.figure(figsize=(30, 10))
        sns.countplot(x="Year", data=data, hue="Genre", order=data['Year'].value_counts().iloc[:5].index)
        plt.xticks(size=16, rotation=90)
        plt.tight_layout()
        plot_filename = "top5_years_genre_count_plot.png"
        plot_filepath = os.path.join(app.config['UPLOAD_FOLDER'], plot_filename)
        plt.savefig(plot_filepath)
        plt.close()
        plot_urls.append(url_for('static', filename='uploads/' + plot_filename))

    # New plot for Global_Sales by Year
    if 'Year' in data.columns and 'Global_Sales' in data.columns:
        data_year = data.groupby(by=['Year'])['Global_Sales'].sum().reset_index()
        plt.figure(figsize=(15, 10))
        sns.barplot(x="Year", y="Global_Sales", data=data_year)
        plt.xticks(rotation=90)
        plt.title('Global Sales by Year')
        plt.tight_layout()
        plot_filename = "global_sales_by_year.png"
        plot_filepath = os.path.join(app.config['UPLOAD_FOLDER'], plot_filename)
        plt.savefig(plot_filepath)
        plt.close()
        plot_urls.append(url_for('static', filename='uploads/' + plot_filename))

    for column in numeric_data.columns:
        insights.append({
            'column': column,
            'mean': float(numeric_data[column].mean()),
            'median': float(numeric_data[column].median()),
            'std': float(numeric_data[column].std()),
            'min': float(numeric_data[column].min()),
            'max': float(numeric_data[column].max())
        })

    plt.figure(figsize=(13, 10))
    sns.heatmap(data.corr(), cmap="Blues", annot=True, linewidth=3)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plot_filename = "correlation_matrix.png"
    plot_filepath = os.path.join(app.config['UPLOAD_FOLDER'], plot_filename)
    plt.savefig(plot_filepath)
    plt.close()
    plot_urls.append(url_for('static', filename='uploads/' + plot_filename))

    if 'NA_Sales' in data.columns and 'EU_Sales' in data.columns:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data['NA_Sales'], label='NA_Sales', shade=True)
        sns.kdeplot(data['EU_Sales'], label='EU_Sales', shade=True)
        plt.title('Distribution of NA_Sales vs EU_Sales')
        plt.legend()
        plt.tight_layout()
        plot_filename = "na_vs_eu_sales.png"
        plot_filepath = os.path.join(app.config['UPLOAD_FOLDER'], plot_filename)
        plt.savefig(plot_filepath)
        plt.close()
        plot_urls.append(url_for('static', filename='uploads/' + plot_filename))

    if 'JP_Sales' in data.columns and 'Other_Sales' in data.columns:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data['JP_Sales'], label='JP_Sales', shade=True)
        sns.kdeplot(data['Other_Sales'], label='Other_Sales', shade=True)
        plt.title('Distribution of JP_Sales vs Other_Sales')
        plt.legend()
        plt.tight_layout()
        plot_filename = "jp_vs_other_sales.png"
        plot_filepath = os.path.join(app.config['UPLOAD_FOLDER'], plot_filename)
        plt.savefig(plot_filepath)
        plt.close()
        plot_urls.append(url_for('static', filename='uploads/' + plot_filename))

    if all(col in data.columns for col in ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']):
        data['Difference_NA_EU'] = data['NA_Sales'] - data['EU_Sales']
        data['Difference_JP_Other'] = data['JP_Sales'] - data['Other_Sales']
        top_sale_reg = data[['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']].sum().reset_index()
        top_sale_reg.columns = ['region', 'sale']
        labels = top_sale_reg['region']
        sizes = top_sale_reg['sale']
        plt.figure(figsize=(10, 8))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
        plt.title('Sales Distribution by Region')
        plt.tight_layout()
        plot_filename = "sales_distribution_by_region.png"
        plot_filepath = os.path.join(app.config['UPLOAD_FOLDER'], plot_filename)
        plt.savefig(plot_filepath)
        plt.close()
        plot_urls.append(url_for('static', filename='uploads/' + plot_filename))

    return plot_urls, summary_stats, insights, dataframe_html

def generate_summary_plot(summary_stats):
    plt.figure(figsize=(12, 8))
    all_summary_stats = pd.DataFrame()
    for filename, stats in summary_stats.items():
        df = pd.DataFrame(stats)
        df['file'] = filename
        all_summary_stats = pd.concat([all_summary_stats, df])

    all_summary_stats = all_summary_stats.reset_index().rename(columns={'index': 'stat'})
    summary_pivot = all_summary_stats.pivot(index='stat', columns='file', values='mean')
    summary_pivot.plot(kind='bar', figsize=(15, 8))
    plt.title("Summary Statistics - Mean Values")
    plt.xlabel("Statistic")
    plt.ylabel("Value")
    plt.legend(title="File", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_filename = "summary_statistics_bar_chart.png"
    plot_filepath = os.path.join(app.config['UPLOAD_FOLDER'], plot_filename)
    plt.savefig(plot_filepath)
    plt.close()
    return url_for('static', filename='uploads/' + plot_filename)

def generate_insights_plot(insights):
    plt.figure(figsize=(12, 8))
    df_insights = pd.DataFrame(insights)
    insights_melted = df_insights.melt(id_vars=["column"], value_vars=["mean", "median", "std", "min", "max"])
    insights_pivot = insights_melted.pivot(index='column', columns='variable', values='value')
    insights_pivot.plot(kind='area', figsize=(15, 8), stacked=False)
    plt.title("Insights")
    plt.xlabel("Column")
    plt.ylabel("Value")
    plt.legend(title="Statistic", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plot_filename = "insights_area_chart.png"
    plot_filepath = os.path.join(app.config['UPLOAD_FOLDER'], plot_filename)
    plt.savefig(plot_filepath)
    plt.close()
    return url_for('static', filename='uploads/' + plot_filename)

def save_insights_to_db(filename, insights):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('CREATE TABLE IF NOT EXISTS insights (filename VARCHAR(255), column_name VARCHAR(255), mean FLOAT, median FLOAT, std FLOAT, min FLOAT, max FLOAT)')

    for insight in insights:
        cursor.execute(
            'INSERT INTO insights (filename, column_name, mean, median, std, min, max) VALUES (%s, %s, %s, %s, %s, %s, %s)',
            (filename, insight['column'], insight['mean'], insight['median'], insight['std'], insight['min'], insight['max'])
        )

    conn.commit()
    cursor.close()
    conn.close()

if __name__ == '__main__':
    app.run(debug=True)
