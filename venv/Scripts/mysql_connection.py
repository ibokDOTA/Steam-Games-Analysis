import mysql.connector

try:
    # Establish the connection
    conn = mysql.connector.connect(
        host="127.0.0.1",
        user="root",
        password="12345",
        database="steam games"
    )

    if conn.is_connected():
        print("Successfully connected to the database")

    # Create a cursor object
    cursor = conn.cursor()

    # Execute a query
    query = "SELECT * FROM your_table"
    cursor.execute(query)

    # Fetch and print all rows from the executed query
    rows = cursor.fetchall()
    for row in rows:
        print(row)

except mysql.connector.Error as err:
    print(f"Error: {err}")

finally:
    # Close the cursor and connection
    if cursor:
        cursor.close()
    if conn:
        conn.close()
