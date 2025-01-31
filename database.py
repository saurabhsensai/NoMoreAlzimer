# database.py
import sqlite3
import pandas as pd

def init_db():
    conn = sqlite3.connect('patient_data.db')
    c = conn.cursor()
    
    # Create table (same as before)
    c.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            PTID TEXT PRIMARY KEY,
            NAME TEXT,
            C2N_plasma_ptau217_ratio REAL,
            C2N_plasma_Abeta42 REAL,
            C2N_plasma_Abeta42_Abeta40 REAL,
            C2N_plasma_ptau217 REAL,
            Roche_plasma_GFAP REAL,
            C2N_plasma_nptau217 REAL,
            Roche_plasma_NfL REAL,
            DX TEXT,
            AGE INTEGER,
            MMSE REAL,
            ADAS11 REAL,
            ADAS13 REAL,
            FAQ REAL,
            APOE4 INTEGER,
            PTETHCAT TEXT,
            PTGENDER TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def add_patient(data):
    conn = sqlite3.connect('patient_data.db')
    c = conn.cursor()
    
    columns = ', '.join(data.keys())
    placeholders = ':' + ', :'.join(data.keys())
    query = f'INSERT INTO patients ({columns}) VALUES ({placeholders})'
    
    try:
        c.execute(query, data)
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_all_patients():
    conn = sqlite3.connect('patient_data.db')
    query = "SELECT * FROM patients"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_patient(ptid):
    conn = sqlite3.connect('patient_data.db')
    query = "SELECT * FROM patients WHERE PTID = ?"
    df = pd.read_sql_query(query, conn, params=(ptid,))
    conn.close()
    return df

# New function to delete a patient
def delete_patient(ptid):
    conn = sqlite3.connect('patient_data.db')
    c = conn.cursor()
    
    try:
        print(f"[DEBUG] Attempting to delete PTID: {ptid}")  # Debugging line
        c.execute("DELETE FROM patients WHERE PTID = ?", (ptid,))
        conn.commit()
        success = c.rowcount > 0
        print(f"[DEBUG] Deletion successful: {success}")  # Debugging line
        return success
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        return False
    finally:
        conn.close()



