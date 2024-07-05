from setuptools import find_packages, setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(
    name="fraud_detection",
    version="0.0.1",
    description="Online payment fraud detection",
    author="Weena",
    author_email="jatmaurice@gmail.com",
    url="https://github.com/sylvain12/fraud-detection-ml",
    install_requires=requirements,
    packages=find_packages(),
    package_dir={"": "src"},
    test_suite="tests",
    # include_package_data: to install data from MANIFEST.in
    include_package_data=True,
    zip_safe=False,
)
