MAP_DICT = {
    "Calculation": """Table Structure:
[
    {"Loan ID": 101, "Principal Amount (USD)": "10,000", "Interest Rate (%)": 5, "Term (Years)": 3, "Total Payable Amount (USD)": "10,500"},
    {"Loan ID": 102, "Principal Amount (USD)": "15,000", "Interest Rate (%)": 7, "Term (Years)": 5, "Total Payable Amount (USD)": "19,250"},
    {"Loan ID": 103, "Principal Amount (USD)": "20,000", "Interest Rate (%)": 6, "Term (Years)": 4, "Total Payable Amount (USD)": "23,000"}
]
Anomaly:
    - For Loan ID 101: The recorded value is 10,500, which is incorrect.
    - But, 10,000 + (10,000 x 0.05 x 3) is 11,500
    - Using the formula for simple interest: Total Payable = Principal + (Principal x Rate x Term)

Table Structure:
[
    {"Business ID": 301, "Revenue (IJSD)": "100,000", "Expense (USD)": "90,000", "Declared profit (USD)": "15,000"},
    {"Business ID": 302, "Revenue (IJSD)": "200,000", "Expense (USD)": "180,000", "Declared profit (USD)": "5,000"},
    {"Business ID": 303, "Revenue (IJSD)": "150,000", "Expense (USD)": "120,000", "Declared profit (USD)": "30,000"}
]
Anomaly:
    - For Business ID 302: The Declared profit is 5,000.
    - But, 200,000 - 180,000 = 20,000
    - Profit is calculated as: Profit = Revenue – Expense""",

    "Data_Consistency": """Table Structure:
[
    {"Customer ID": 101, "Name": "John Doe", "Phone Number": "(123) 456-7890", "Email": "john.doe@email.com"},
    {"Customer ID": 102, "Name": "Jane Smith", "Phone Number": "123-456-7890", "Email": ""},
    {"Customer ID": 103, "Name": "Robert Johnson", "Phone Number": "1234567890", "Email": "rob.johnson@email.com"},
    {"Customer ID": 104, "Name": "John Doe", "Phone Number": "+1 123 456 7890", "Email": "john.doe@email.com"}
]
Anomaly:
    - The Phone Number column uses inconsistent formatting.
        • Customer ID 101 uses parentheses and a dash.
        • Customer ID 102 uses dashes only.
        • Customer ID 103 is unformatted.
        • Customer ID 104 uses international format.
    - Name for Customer ID 104 is duplicated (same as ID 101), indicating a potential duplicate entry.
    - Email for Customer ID 104 is also the same as for Customer ID 101.
    - Email is missing for Customer ID 102.

Table Structure:
[
    {"Product ID": "P001", "Product Name": "iPhone 14", "Category": "Electronics", "Price (USD)": 799, "Supplier": "Samsung"},
    {"Product ID": "P002", "Product Name": "iPhone 14", "Category": "Electronics", "Price (USD)": 750, "Supplier": "Samsung"},
    {"Product ID": "P003", "Product Name": "Samsung Galaxy S23", "Category": "Electronics", "Price (USD)": 799, "Supplier": "Samsung"},
    {"Product ID": "P004", "Product Name": "Samsung Galaxy S23", "Category": "Electronics", "Price (USD)": 780, "Supplier": "Sa rnsu ng"}
]
Anomaly:
    - Product ID P001 and P002 are the same product (iPhone 14) but have different prices (799 USD vs. 750 USD).
    - Product ID P003 and P004 are also the same (Samsung Galaxy S23) but priced at 799 USD and 780 USD respectively.
    - Supplier name for Product ID P004 is misspelled as "Sa rnsu ng".
""",

    "Factual_Anomaly": """Table Structure:
[
    {"Employee ID": 201, "Position": "Intern", "Salary (USD)": 80000},
    {"Employee ID": 202, "Position": "Manager", "Salary (USD)": 50000}
]
Anomaly:
    - Employee ID 201 is marked as an "Intern" but has a higher salary (80,000 USD) than Employee ID 202, a "Manager" earning 50,000 USD.
    - This contradicts typical organizational pay structures where managers generally earn more than interns.

Table Structure:
[
    {"Product ID": "P001", "Product Name": "iPhone 14", "Category": "Electronics", "Price (USD)": 5},
    {"Product ID": "P002", "Product Name": "Banana", "Category": "Grocery", "Price (USD)": 1000}
]
Anomaly:
    - The price of Product ID P001 (iPhone 14) is unrealistically low at 5 USD.
    - The price of Product ID P002 (Banana) is abnormally high at 1,000 USD.
    - These values suggest possible data entry or labeling errors, as they fall outside expected pricing norms for these items.
""",
    "Logical_Anomaly": """Table Structure:
[
    {"Location ID": 101, "City": "Sydney", "Recorded Temperature (°C)": -20, "Date": "2024-07-15", "Hemisphere": "Southern"},
    {"Location ID": 102, "City": "Toronto", "Recorded Temperature (°C)": 40, "Date": "2024-12-22", "Hemisphere": "Northern"}
]
Anomaly:
    - For Location ID 101: Sydney is in the Southern Hemisphere, and July corresponds to winter. A temperature of -20°C is extremely rare and unlikely in this region.
    - For Location ID 102: Toronto is in the Northern Hemisphere, and December is wintertime. A temperature of 40°C is implausible during this season.

Table Structure:
[
    {"Person ID": 301, "Name": "George", "Birth Year": 1820, "Current Year": 2024, "Age": 204},
    {"Person ID": 302, "Name": "Hannah", "Birth Year": 1900, "Current Year": 2024, "Age": 124}
]
Anomaly:
    - For Person ID 301: The recorded age is 204 years, which exceeds the oldest verified human age of 122 years, indicating a likely error.
    - For Person ID 302: An age of 124 also surpasses the maximum documented lifespan, suggesting potential data inaccuracy.
""",
    "Normalization":"""Table Structure:
[
    {"Student ID": "S001", "Student Name": "Alice Green", "Course ID": "C001", "Course Name": "Math 101", "Instructor Name": "Dr. Smith", "Instructor Email": "dr.smith@school.com", "Grade": "A"},
    {"Student ID": "S002", "Student Name": "Bob White", "Course ID": "C002", "Course Name": "History 101", "Instructor Name": "Dr. Johnson", "Instructor Email": "drjohnson@school.com", "Grade": "B"},
    {"Student ID": "S003", "Student Name": "Alice Green", "Course ID": "C003", "Course Name": "Chemistry 101", "Instructor Name": "Dr. Brown", "Instructor Email": "dr.brown@school.com", "Grade": "A"}
]
Anomaly:
    - Transitive Dependency (3NF Violation): Instructor Email depends on Instructor Name, which depends on Course ID.
    - This violates the Third Normal Form (3NF) because non-key attributes (Instructor Email) transitively depend on the primary key (Course ID).""",
    "Security": """Table Structure:
[
    {"User ID": 101, "Name": "John Smith", "Role": "Admin", "Last Login": "2024-03-01", "Account Status": "Active", "Permissions": "Full Access"},
    {"User ID": 102, "Name": "Jane Doe", "Role": "User", "Last Login": "2024-03-03", "Account Status": "Active", "Permissions": "Read-Only"},
    {"User ID": 103, "Name": "Alice Brown", "Role": "Admin", "Last Login": "2024-02-15", "Account Status": "Suspended", "Permissions": "Read-Only"},
    {"User ID": 104, "Name": "Bob Johnson", "Role": "User", "Last Login": "2024-03-05", "Account Status": "Active", "Permissions": ""}
]
Anomaly:
    - Alice Brown (User ID 103) is marked as "Admin" but has a "Suspended" account status.
    - This presents a **security risk** since suspended users should not retain admin privileges or access.
    - Permissions mismatch: Admin role should align with "Full Access", but she only has "Read-Only".

Table Structure:
[
    {"Employee ID": 101, "Username": "johnsmith", "Login Time": "2024-03-01 09:00 AM", "IP Address": "192.168.1.101", "Location": "New York, USA", "Device Type": "Desktop"},
    {"Employee ID": 102, "Username": "janedoe", "Login Time": "20240-02 08:30 AM", "IP Address": "192.168.1.102", "Location": "California, USA", "Device Type": "Laptop"},
    {"Employee ID": 103, "Username": "alicebrown", "Login Time": "20240-01 11:00 AM", "IP Address": "10.0.0.1", "Location": "Unknown", "Device Type": "Mobile"},
    {"Employee ID": 104, "Username": "bobjohnson", "Login Time": "2024-03-03 02:15 PM", "IP Address": null, "Location": "Unknown", "Device Type": null}
]
Anomaly:
    - Employee ID 104 has **NULL** values for both IP Address and Device Type, which are critical for auditing login attempts.
    - Alice Brown (Employee ID 103) logged in from an **unknown location**, which may indicate suspicious activity, especially given her suspended status in the access control table.
    - Date Format Issue: Login Time entries for Employee ID 102 and 103 use an incorrect year format ("20240").""",
    
    "Temporal": """Table Structure:
[
    {"Employee ID": "EOOI", "Work Date": "2024-03-01", "Check-In Time": "09:00 AM", "Check-Out Time": "05:00 PM", "Total Hours": 9},
    {"Employee ID": "E002", "Work Date": "2024-03-01", "Check-In Time": "08:00 AM", "Check-Out Time": "04:00 PM", "Total Hours": 6}
]
Anomaly:
    - For Employee ID EOOI, the Total Hours (9) is inconsistent with the duration between Check-In (09:00 AM) and Check-Out (05:00 PM), which is 8 hours.
    - For Employee ID E002, the Total Hours (6) is inconsistent with the recorded times (Check-In at 08:00 AM and Check-Out at 04:00 PM), which should equal 8 hours.

Table Structure: 
[
    {"Flight ID": "FOOI", "Departure Time": "09:00 AM", "Arrival Time": "08:30 AM", "Duration (hours)": 1.5},
    {"Flight ID": "F002", "Departure Time": "10:00 AM", "Arrival Time": "01:00 PM", "Duration (hours)": 3}
]
Anomaly:
    - For Flight ID FOOI, the Arrival Time (08:30 AM) is earlier than the Departure Time (09:00 AM), which is a logical error.
    - The calculated Duration (1.5 hours) does not align with the given Departure and Arrival times (which suggests a negative duration).
    - This is a clear case of data inconsistency and requires correction in the schedule.
""",
    "Value": """Table Structure: 
[
    {"Transaction ID": "TXNOOI", "Date": "2024-11-01", "Amount": "$50.00", "Customer Info": {"Name": "Alice"}, "Payment Status": "Completed", "Notes": "Unusual outlier"},
    {"Transaction ID": "TXN002", "Date": "2024-11-02", "Amount": "$20,000.00", "Customer Info": {"Name": "Bob"}, "Payment Status": "Completed", "Notes": "Unusually high value"},
    {"Transaction ID": "TXN003", "Date": "2024-11-03", "Amount": "-$30.00", "Customer Info": {"Name": "Charlie"}, "Payment Status": "Failed", "Notes": "Negative value"}
]
Anomaly:
    - For Transaction ID TXN002, the Amount of $20,000.00 is unusually high compared to typical transaction amounts (typically between $10-$200).
    - For Transaction ID TXN003, the Amount of -$30.00 is negative, which is logically invalid for a payment transaction.
    - For TXNOOI, the note mentions an "unusual outlier," indicating an anomaly in this transaction as well.

Table Structure: 
[
    {"User ID": 10001, "Name": "John Doe", "Age": 25, "Registration Date": "2024-01-01"},
    {"User ID": 10002, "Name": "Jane Smith", "Age": 120, "Registration Date": "2024-01-01"},
    {"User ID": 10003, "Name": "Emily Johnson", "Age": -5, "Registration Date": "2024-01-01"}
]
Anomaly:
    - For User ID 10002, the Age (120) is implausibly high and exceeds the typical human lifespan.
    - For User ID 10003, the Age (-5) is negative, which is logically impossible.
"""
}
