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


