import sqlite3

from sqlalchemy import create_engine

conn = sqlite3.connect("multiplier.db")
conn.execute(""CREATE TABLE if not exists multiplier
        (domain      CHAR(50),
         low      REAL,
         high      REAL);""")
conn.close()
dbname = "sqlite:///" + prop + "_" + domain + str(i) + ".db"
diskengine = create_engine(dbname)
df.to_sql("scores", diskengine, if_exists="replace")
