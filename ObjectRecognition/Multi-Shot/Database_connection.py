import sqlite3
import numpy as np
import io

conn = sqlite3.connect('Database.db')
conn.text_factory = sqlite3.Binary
c = conn.cursor()

def insert_classification_label_to_database(classification, label):
    # Insert a row of data
    query = "INSERT INTO Vector_Recognition(classification, label) VALUES(?, ?)"
    c.execute(query, (sqlite3.Binary(classification), label))

    # Save (commit) the changes
    conn.commit()


test = np.array(["1", "2"])
insert_classification_label_to_database(test, "877")

def select_classification_label_to_database():
    # Select a row of data
    query = "SELECT classification FROM Vector_Recognition"
    results = c.execute(query)

    for row in results:
        print(row[0])

    # Save (commit) the changes
    conn.commit()

    # Close the connection
    conn.close()

select_classification_label_to_database()