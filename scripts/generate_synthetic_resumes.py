import csv, random
from faker import Faker

fake = Faker()  # Faker generates realistic fake data (names, skills, etc.)

# Number of synthetic resumes
N = 2000

# Protected attributes
genders = ['male','female','non-binary','unknown']
ethnicities = ['groupA','groupB','groupC','unknown']

# Open CSV file for writing
with open('data/synthetic_resumes.csv','w', newline='') as f:
    w = csv.writer(f)
    # Write header row
    w.writerow(['candidate_id','gender','ethnicity','education','years_experience','skills','label'])

    for i in range(N):
        # Randomly assign gender and ethnicity with some weights
        g = random.choices(genders, weights=[0.45,0.45,0.05,0.05])[0]
        e = random.choices(ethnicities, weights=[0.6,0.25,0.1,0.05])[0]

        # Random education level
        edu = random.choice(['bachelors','masters','phd'])
        # Random years of experience
        yrs = random.randint(0,20)
        # Random skills (4 fake words joined with ;)
        skills = ";".join(fake.words(nb=4))

        # Simulate biased hiring label
        base = 0.2 + 0.03*yrs + (0.1 if edu=='masters' else 0) + (0.15 if edu=='phd' else 0)
        if g=='female' and e=='groupB':  # introduce bias
            base -= 0.07
        # Assign label probabilistically
        label = 1 if random.random() < base else 0

        # Write row to CSV
        w.writerow([i,g,e,edu,yrs,skills,label])
