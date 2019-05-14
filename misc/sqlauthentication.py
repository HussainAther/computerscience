import mysql.connector 
import hashlib
 
import sys	 
import random	

"""
Connect to a MySQL database (connect_db), create user/password records in the following table (create_user), and
authenticate login requests against the table (authenticate_user).
"""

DB_HOST = "localhost"	 
DB_USER = "devel" 
DB_PASS = "devel"	 
DB_NAME = "test"	 
 
def connect_db():	 
    """
    Try to connect DB and return DB instance, if not, return False.
    """
    try:	 
        return mysql.connector.connect(host=DB_HOST, user=DB_USER, passwd=DB_PASS, db=DB_NAME)	 
    except:	 
        return False	

def create_user(username, passwd): 
    """
    If user was successfully created, returns its ID; returns None on error.
    """
    db = connect_db()	 
    if not db:	 
        print("Can't connect MySQL!")
        return None
    cursor = db.cursor()	 
    salt = randomValue(16)	 	 
    passwd_md5 = hashlib.md5(salt+passwd).hexdigest()	  
