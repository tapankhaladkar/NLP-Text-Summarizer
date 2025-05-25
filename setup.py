import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.0"

REPO_NAME = "NLP-Text-Summarizer"
AUTHOR_USER_NAME = "tapankhaladkar"
SRC_REPO = 'textsummarizer'
AUTHOR_EMAIL = "tapankhaladkar@gmail.com"   

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A Python library for text summarization using NLP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",

    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)
 #   classifiers=[
  ##     "License :: OSI Approved :: MIT License",
    ##    "Operating System :: OS Independent",
