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
