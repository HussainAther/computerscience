
"""
"""

print("Plotting the generated distribution...")
values = extract(g_fake_data)
print(" Values: %s" % (str(values)))
plt.hist(values, bins=50)
plt.xlabel('Value')
plt.ylabel('Count')
plt.title("Histogram of Generated Distribution")
plt.grid(True)
plt.show()
