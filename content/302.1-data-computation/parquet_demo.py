import marimo

__generated_with = "0.16.2"
app = marimo.App()


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    from pyspark.sql import SparkSession
    from pyspark.sql.types import (
        StructType,
        StructField,
        IntegerType,
        StringType,
        DoubleType,
    )
    import os
    import time
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pyspark.sql.functions import lit

    # Unified color scheme for consistent styling
    COLORS = {
        'primary': '#2563eb',      # blue-600
        'secondary': '#7c3aed',    # violet-600
        'accent': '#0d9488',       # teal-600
        'success': '#16a34a',      # green-600
        'warning': '#d97706',      # amber-600
        'danger': '#dc2626',       # red-600
        'info': '#0ea5e9',         # sky-500
        'background': '#f8fafc',   # slate-50
        'card': '#ffffff',         # white
        'text': '#1e293b',         # slate-800
        'text_secondary': '#64748b' # slate-500
    }
    return (
        COLORS,
        IntegerType,
        SparkSession,
        StringType,
        StructField,
        StructType,
        lit,
        mo,
        np,
        os,
        pd,
        plt,
        time,
    )


@app.cell(hide_code=True)
def _(COLORS, mo):
    mo.md(
        rf"""
    <div style="background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['secondary']} 100%); padding: 30px; border-radius: 15px; color: white; box-shadow: 0 10px 25px rgba(0,0,0,0.2);">
    <h1 style="margin-top: 0; font-size: 2.5em; text-align: center;">📊 Parquet vs CSV Demonstration with PySpark</h1>
    <p style="font-size: 1.2em; text-align: center; max-width: 800px; margin: 0 auto;">
    This interactive notebook demonstrates the key differences between <strong>CSV</strong> and <strong>Parquet</strong> file formats using <strong>PySpark</strong>. 
    We'll generate sample data, compare performance metrics, and explore schema evolution capabilities.
    </p>
    </div>

    <div style="display: flex; flex-wrap: wrap; gap: 20px; margin: 30px 0;">
        <div style="flex: 1; min-width: 300px; background-color: {COLORS['card']}; border-radius: 12px; padding: 25px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
            <h2 style="color: {COLORS['primary']}; margin-top: 0;">🎯 Learning Objectives</h2>
            <ul style="padding-left: 20px;">
                <li>Understand CSV vs Parquet format differences</li>
                <li>Compare read/write performance and file sizes</li>
                <li>Explore columnar storage benefits in Parquet</li>
                <li>Learn about schema evolution in big data workflows</li>
            </ul>
        </div>

        <div style="flex: 1; min-width: 300px; background-color: {COLORS['card']}; border-radius: 12px; padding: 25px; box-shadow: 0 4px 12px rgba(0,0,0,0.08);">
            <h2 style="color: {COLORS['secondary']}; margin-top: 0;">🔬 How Parquet Works</h2>
            <ol style="padding-left: 20px;">
                <li><strong>Columnar Storage</strong> 📈: Stores data by columns for efficient queries</li>
                <li><strong>Compression</strong> 💾: Built-in compression reduces file sizes significantly</li>
                <li><strong>Schema Evolution</strong> 🔄: Supports adding columns without rewriting files</li>
                <li><strong>Spark Integration</strong> ⚡: Native support for distributed processing</li>
            </ol>
        </div>
    </div>

    <div style="background-color: {COLORS['card']}; border-radius: 12px; padding: 25px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); margin: 30px 0;">
        <h2 style="color: {COLORS['accent']}; margin-top: 0;">🛠️ Key Components of This Demo</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px;">
            <div style="padding: 15px; background-color: rgba(37, 99, 235, 0.05); border-radius: 8px; border-left: 4px solid {COLORS['primary']};">
                <strong>PySpark</strong>: Distributed data processing engine for big data
            </div>
            <div style="padding: 15px; background-color: rgba(124, 58, 237, 0.05); border-radius: 8px; border-left: 4px solid {COLORS['secondary']};">
                <strong>Performance Metrics</strong>: Timing write/read operations and file sizes
            </div>
            <div style="padding: 15px; background-color: rgba(13, 148, 136, 0.05); border-radius: 8px; border-left: 4px solid {COLORS['accent']};">
                <strong>Data Generation</strong>: Synthetic dataset with 100k+ rows for realistic testing
            </div>
            <div style="padding: 15px; background-color: rgba(22, 163, 74, 0.05); border-radius: 8px; border-left: 4px solid {COLORS['success']};">
                <strong>Visualizations</strong>: Charts comparing formats and query results
            </div>
            <div style="padding: 15px; background-color: rgba(217, 119, 6, 0.05); border-radius: 8px; border-left: 4px solid {COLORS['warning']};">
                <strong>Schema Evolution</strong>: Demonstrate adding columns to Parquet files
            </div>
        </div>
    </div>
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 📈 Workflow Visualization

    Below is a visualization of the Parquet vs CSV workflow we'll be implementing:
    """
    )
    return


@app.cell(hide_code=True)
def _(COLORS, mo):
    # Create a diagram of the workflow
    workflow_diagram = mo.mermaid(
        f"""
        flowchart TD
            A[Generate Sample Data<br/>Pandas DataFrame] --> B[Define Schema<br/>StructType]
            B --> C[Create Spark DataFrame]
            C --> D[Write to CSV<br/>Text-based, Row-major]
            C --> E[Write to Parquet<br/>Columnar, Compressed]
            D --> F[Read from CSV<br/>Parse Headers]
            E --> G[Read from Parquet<br/>Schema Inference]
            F --> H[Simple Computations<br/>Aggregations, Filters]
            G --> I[Simple Computations<br/>Faster Column Access]
            H --> J[Schema Evolution<br/>Add Columns - Manual for CSV]
            I --> K[Schema Evolution<br/>Native Support in Parquet]
            J --> L[Compare Performance<br/>Timing & Sizes]
            K --> L

            style A fill:{COLORS['primary']}20,stroke:{COLORS['primary']},stroke-width:2px
            style B fill:{COLORS['secondary']}20,stroke:{COLORS['secondary']},stroke-width:2px
            style C fill:{COLORS['info']}20,stroke:{COLORS['info']},stroke-width:2px
            style D fill:{COLORS['warning']}20,stroke:{COLORS['warning']},stroke-width:2px
            style E fill:{COLORS['success']}20,stroke:{COLORS['success']},stroke-width:2px
            style F fill:{COLORS['danger']}20,stroke:{COLORS['danger']},stroke-width:2px
            style G fill:{COLORS['accent']}20,stroke:{COLORS['accent']},stroke-width:2px
            style H fill:{COLORS['primary']}20,stroke:{COLORS['primary']},stroke-width:2px
            style I fill:{COLORS['secondary']}20,stroke:{COLORS['secondary']},stroke-width:2px
            style J fill:{COLORS['warning']}20,stroke:{COLORS['warning']},stroke-width:2px
            style K fill:{COLORS['success']}20,stroke:{COLORS['success']},stroke-width:2px
            style L fill:{COLORS['info']}40,stroke:{COLORS['info']},stroke-width:2px
        """
    ).center()

    workflow_diagram
    return


@app.cell(hide_code=True)
def _(SparkSession, mo):
    import logging
    import warnings
    warnings.filterwarnings("ignore")
    logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)
    logging.getLogger("pyspark").setLevel(logging.ERROR)
    logging.getLogger("pyspark.sql").setLevel(logging.ERROR)


    # Initialize Spark Session
    spark = (
        SparkSession.builder.master("local[1]")
        .appName("ParquetVsCSV")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    mo.md("SparkSession initialized.")
    return (spark,)


@app.cell(hide_code=True)
def _(COLORS, mo):
    mo.md(
        rf"""
    <div style="background: linear-gradient(135deg, {COLORS['secondary']}10 0%, {COLORS['primary']}10 100%); border-left: 6px solid {COLORS['secondary']}; padding: 25px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
    <h2 style="color: {COLORS['secondary']}; margin-top: 0;">⚙️ Demo Parameters</h2>

    <p>Adjust the parameters to control the dataset size and observe performance differences:</p>

    <h3 style="color: {COLORS['primary']};">📊 Dataset Parameters</h3>
    <ul>
    <li><strong>Number of Rows</strong> 📈: Size of the generated dataset. Larger datasets highlight format differences more clearly.</li>
    </ul>
    </div>
    """
    )
    return


@app.cell(hide_code=True)
def _(COLORS, mo):
    mo.md(
        rf"""
    <div style="background: linear-gradient(135deg, {COLORS['success']}10 0%, {COLORS['accent']}10 100%); border-left: 6px solid {COLORS['success']}; padding: 25px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
    <h2 style="color: {COLORS['success']}; margin-top: 0;">▶️ Execute the Demo</h2>

    <p>Click the button below to run the full Parquet vs CSV comparison with your parameters:</p>

    <h3 style="color: {COLORS['accent']};">🔄 Execution Flow</h3>
    <ol>
    <li>Define schema and create Spark DataFrame from generated data</li>
    <li>Perform write/read operations for both CSV and Parquet formats</li>
    <li>Measure and compare timing and file sizes</li>
    <li>Run computations, filters, and schema evolution</li>
    <li>Generate visualizations and performance metrics</li>
    <li>Save results to demo_data directory</li>
    </ol>
    </div>
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    # Create interactive slider for number of rows
    n_rows_slider = mo.ui.slider(
        start=10000,
        stop=500000,
        step=10000,
        value=100000,
        label="📊 Number of Rows",
        show_value=True
    )
    return (n_rows_slider,)


@app.cell(hide_code=True)
def _(mo):
    run_button = mo.ui.run_button(
        label="🚀 Run Parquet vs CSV Demo"
    )
    return (run_button,)


@app.cell(hide_code=True)
def _(COLORS, mo, n_rows_slider):
    mo.md(rf"""
    <div style="background: linear-gradient(135deg, {COLORS['secondary']}10 0%, {COLORS['accent']}10 100%); border-left: 6px solid {COLORS['secondary']}; padding: 20px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); margin: 25px 0;">
    <h3 style="color: {COLORS['secondary']}; margin-top: 0; text-align: center;">🎛️ Adjust Parameters</h3>
    </div>
    """)

    # Display the slider centered
    _controls = mo.center(n_rows_slider)
    _controls
    return


@app.cell(hide_code=True)
def _(mo, run_button):
    # Display the run button centered
    _run_control = mo.center(run_button)
    _run_control
    return


@app.cell(hide_code=True)
def _(
    COLORS,
    IntegerType,
    StringType,
    StructField,
    StructType,
    lit,
    mo,
    n_rows_slider,
    np,
    os,
    pd,
    plt,
    run_button,
    spark,
    time,
):
    if run_button.value:
        print("### ▶️ Executing Parquet vs CSV Demo...")

        print("## Generate Sample Data")

        # Generate rows based on slider
        n_rows = n_rows_slider.value
        user_ids = np.arange(1, n_rows + 1)
        regions = np.random.choice(["North", "South", "East", "West"], n_rows)
        ages = np.random.randint(18, 65, n_rows)
        salaries = np.random.normal(50000, 15000, n_rows).astype(int)

        # Create Pandas DataFrame first for ease
        pdf = pd.DataFrame(
            {"user_id": user_ids, "region": regions, "age": ages, "salary": salaries}
        )

        print(f"Generated dataset with {n_rows:,} rows.")
        print(pdf.head())

        # Define schema
        schema = StructType([
            StructField("user_id", IntegerType(), True),
            StructField("region", StringType(), True),
            StructField("age", IntegerType(), True),
            StructField("salary", IntegerType(), True),
        ])

        # Create Spark DataFrame
        sdf = spark.createDataFrame(pdf, schema)
        sdf = sdf.coalesce(1)  # Reduce to single partition to avoid memory allocation warnings during Parquet write

        # Demo directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        demo_dir = os.path.join(script_dir, "demo_data")
        os.makedirs(demo_dir, exist_ok=True)
        csv_path = os.path.join(demo_dir, "csv")
        parquet_path = os.path.join(demo_dir, "parquet")

        # Write to CSV
        start_time = time.time()
        sdf.write.mode("overwrite").option("header", "true").csv(csv_path)
        csv_write_time = time.time() - start_time

        # Calculate CSV size (sum of part files, exclude _SUCCESS)
        csv_size = sum(os.path.getsize(os.path.join(csv_path, f)) for f in os.listdir(csv_path) if not f.endswith('_SUCCESS'))

        # Write to Parquet
        start_time = time.time()
        sdf.write.mode("overwrite").parquet(parquet_path)
        parquet_write_time = time.time() - start_time

        # Calculate Parquet size
        parquet_size = sum(os.path.getsize(os.path.join(parquet_path, f)) for f in os.listdir(parquet_path) if not f.endswith('_SUCCESS'))

        # Read from CSV
        start_time = time.time()
        df_csv = spark.read.option("header", "true").schema(schema).csv(csv_path)
        csv_read_time = time.time() - start_time

        # Read from Parquet
        start_time = time.time()
        df_parquet = spark.read.parquet(parquet_path)
        parquet_read_time = time.time() - start_time

        # Perform computation: filter and aggregate
        start_time = time.time()
        result_csv = df_csv.filter(df_csv.age > 30).groupBy("region").avg("salary").collect()
        csv_comp_time = time.time() - start_time

        start_time = time.time()
        result_parquet = df_parquet.filter(df_parquet.age > 30).groupBy("region").avg("salary").collect()
        parquet_comp_time = time.time() - start_time

        # Schema Evolution Example
        print("#### Schema Evolution in Parquet")
        sdf_evolved = sdf.withColumn("bonus", lit(1000))
        evolved_path = os.path.join(demo_dir, "parquet_evolved")
        start_time = time.time()
        sdf_evolved.write.mode("overwrite").parquet(evolved_path)
        evolve_write_time = time.time() - start_time

        # Read both original and evolved
        df_evolved = spark.read.parquet(evolved_path)

        print(f"**Original Parquet Schema:**")
        df_parquet.printSchema()
        print(f"**Evolved Parquet Schema (added 'bonus' column):**")
        df_evolved.printSchema()
        print(f"**Evolution Write Time:** {evolve_write_time:.2f}s")

        # Set consistent color palette
        csv_color = COLORS['warning']
        parquet_color = COLORS['success']

        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("📊 Parquet vs CSV Performance Comparison", fontsize=16, color=COLORS['primary'])

        # Write Times
        bars1 = ax1.bar(['CSV', 'Parquet'], [csv_write_time, parquet_write_time], 
                         color=[csv_color, parquet_color], alpha=0.7)
        ax1.set_title('Write Time', fontsize=12)
        ax1.set_ylabel('Time (seconds)')
        ax1.set_ylim(0, max(csv_write_time, parquet_write_time) * 1.1)
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}s', ha='center', va='bottom')

        # Read Times
        bars2 = ax2.bar(['CSV', 'Parquet'], [csv_read_time, parquet_read_time], 
                         color=[csv_color, parquet_color], alpha=0.7)
        ax2.set_title('Read Time', fontsize=12)
        ax2.set_ylabel('Time (seconds)')
        ax2.set_ylim(0, max(csv_read_time, parquet_read_time) * 1.1)
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}s', ha='center', va='bottom')

        # File Sizes
        sizes_mb = [csv_size / (1024*1024), parquet_size / (1024*1024)]
        bars3 = ax3.bar(['CSV', 'Parquet'], sizes_mb, 
                         color=[csv_color, parquet_color], alpha=0.7)
        ax3.set_title('File Size', fontsize=12)
        ax3.set_ylabel('Size (MB)')
        ax3.set_ylim(0, max(sizes_mb) * 1.1)
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}MB', ha='center', va='bottom')

        # Computation Times
        bars4 = ax4.bar(['CSV', 'Parquet'], [csv_comp_time, parquet_comp_time], 
                         color=[csv_color, parquet_color], alpha=0.7)
        ax4.set_title('Computation Time', fontsize=12)
        ax4.set_ylabel('Time (seconds)')
        ax4.set_ylim(0, max(csv_comp_time, parquet_comp_time) * 1.1)
        for bar in bars4:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}s', ha='center', va='bottom')

        plt.tight_layout()

        plt.show()

        # Summary table
        print("### 📈 Performance Summary")
        print("")
        print("| Metric | CSV | Parquet | Improvement |")
        print("|--------|-----|---------|-------------|")
        print(f"| Write Time (s) | {csv_write_time:.2f} | {parquet_write_time:.2f} | {((csv_write_time - parquet_write_time)/csv_write_time * 100):.1f}% faster |")
        print(f"| Read Time (s) | {csv_read_time:.2f} | {parquet_read_time:.2f} | {((csv_read_time - parquet_read_time)/csv_read_time * 100):.1f}% faster |")
        print(f"| File Size (MB) | {sizes_mb[0]:.1f} | {sizes_mb[1]:.1f} | {((sizes_mb[0] - sizes_mb[1])/sizes_mb[0] * 100):.1f}% smaller |")
        print(f"| Comp Time (s) | {csv_comp_time:.4f} | {parquet_comp_time:.4f} | {((csv_comp_time - parquet_comp_time)/csv_comp_time * 100):.1f}% faster |")
        print("")
        print("**Parquet shows significant advantages in read/write performance and storage efficiency!** 🚀")
        print("")
        print("Demo completed successfully. You can adjust parameters and click the button again to re-run.")
    else:
        mo.md("Click the button to run the demo.")
    return


if __name__ == "__main__":
    app.run()
