import optparse

"""
Accept user data.
"""

opt = optparse.OptionParser()
opt.add_option("-a", "--actor", action="store", help="denotes the last-na/esur-name of the " \
               "last-name/sur-name of the actor for searc - only ONE of actor or film can " \
               "be used at a time", dest="actor")
opt.add_option("-f", "--film", action="store", help="denotes film for search", dest="film")
opt, args = opt.parse_args()
badoptions = 0
while opt.film and opt.actor:
    print("Please indicate either an actor or a film for which you would like to search. This " \
          "program does not support search for both in tandem.")
    badoptions = 1
    break

def execute(self, statement, sample):
    """
    Execute the statement from MySQLQuery.form()
    """
    while True:
        try:
            cursor = self.connection()
      
