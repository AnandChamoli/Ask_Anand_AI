def analyze_job_description(job_text):

    anand_skills = [
        "python",
        "sql",
        "power bi",
        "tableau",
        "machine learning",
        "data analysis",
        "business analysis",
        "business analyst",
        "customer analytics",
        "sales analytics",
        "predictive modeling",
        "data visualization",
        "dashboard",
        "pyspark",
        "gcp",
        "google cloud",
        "churn prediction",
        "customer segmentation",
        "k-means",
        "random forest",
        "logistic regression",
        "decision tree",
        "forecasting",
        "arima",
        "excel",
        "stakeholder management",
        "business transformation"
    ]

    job_text_lower = job_text.lower()

    jd_skills_found = []
    matched_skills = []
    missing_skills = []

    # Check every skill against JD
    for skill in anand_skills:

        if skill in job_text_lower:
            jd_skills_found.append(skill.title())
            matched_skills.append(skill.title())

    # Optional common external skills
    external_skills = [
        "sap",
        "snowflake",
        "azure",
        "aws",
        "oracle",
        "jira",
        "salesforce"
    ]

    for skill in external_skills:

        if skill in job_text_lower:
            jd_skills_found.append(skill.title())

            if skill not in anand_skills:
                missing_skills.append(skill.title())

    # Score calculation
    if len(jd_skills_found) > 0:
        score = int((len(matched_skills) / len(jd_skills_found)) * 100)
    else:
        score = 0

    return score, matched_skills, missing_skills