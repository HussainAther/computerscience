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
    try:	 
        cursor.execute("INSERT INTO users (`username`, `pass_salt`, `pass_md5`) VALUES (%s, %s, %s)", (username, salt, passwd_md5)) 
        cursor.execute("SELECT userid FROM users WHERE username=%s", (username,) ) 
        id = cursor.fetchone()
        db.commit()
        cursor.close()
        db.close()
        return id[0]	 
    except:	 
        print("Username was already taken. Please select another") 
        return None

 def authenticate_user(username, passwd):
    """
    Authenticate.
    """
    db = connect_db()	 
    if not db:	 
        print("Can't connect MySQL!")
        return False
    cursor = db.cursor()	 
    row = cursor.fetchone()
    cursor.close()
    db.close()
    if row is None: # username not found
        return False
    salt = row[0]
    correct_md5 = row[1]
    tried_md5 = hashlib.md5(salt+passwd).hexdigest()
    return correct_md5 == tried_md5
 
def randomValue(length):	 
    """
    Creates random value with given length.
    """
    salt_chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
    return "".join(random.choice(salt_chars) for x in range(length)) 
