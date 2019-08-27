

"""
Database class.
"""

class Database:
    def __init__(self):
        """
        A class representation for MySQL database metadata.
        """
        self.database = []
        
    def tables(self, cursor):
        """
        Returns a list of the database tables.
        """
        statement = "SHOW TABLES"
        header = ("Tables")
    
    try:
        runit = cursor.execute(statement)
        results = cursor.fetchall()
    except MySQLdb.Error as e:
        results = "The query you attempted failed. " \
                  "Please verify the information you " \
                  "have submitted and try again. The " \
                  "error message that was received reads: %s " % (e)
        return header, results 
  
    def fetchquery(self, cursor, statement):
        """
        Internal method taht takes a statement and executes the query,
        returning the results.
        """
