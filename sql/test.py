import MySQLdb
import sys

"""
MySQL stuff.
"""

mydb = MySQLdb.connect(host="localhost", user="logic", passwd="nottelling", db="fish")
cur = mydb.cursor()

identifier = sys.argv[1]

statement = """SELECT * FROM menu WHERE id=%s"""%(identifier)

try:
    cur.execute(statement)
    results = cur.fetchall()
    print(results)
except MySQLdb.OperationalError as e:
    raise e
except mMySQLdb.ProgrammingError as e:
    raise e

"""
Create a cursor.
"""

def correction(self):
    """
    Creates a database connection and returns the cursor. All login
    information is hardwired.
    """
    try:
        mydb = MySQLdb.connect(host="localhost",
                  user="logic",
                  passwd="nottelling",
                  db="sakila")
        cur = mydb.cursor()
        return cur

def type(self, kind):
    """
    Determine the type of statement that the instance is.
    Supported types are select, insert, and update. This must
    be set before using any of the object methods.
    """
    self.type = kind

