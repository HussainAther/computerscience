import sqlite3

from sqlalchemy import create_engine

conn = sqlite3.connect("multiplier.db")
conn.execute(""CREATE TABLE if not exists multiplier
        (domain      CHAR(50),
         low      REAL,
         high      REAL);""")
conn.close()
