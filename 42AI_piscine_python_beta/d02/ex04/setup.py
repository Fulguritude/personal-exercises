import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
	name='ai42',
	version='1.0.0',
#	scripts=['42ai'],
	author="Tristan Duquesne",
	author_email="tduquesn@student.42.fr",
	description="A small Python Piscine package example",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="",
	packages=setuptools.find_packages(),
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
	],
)