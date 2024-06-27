from setuptools import setup, find_namespace_packages
import platform

DEPENDENCY_LINKS = []
if platform.system() == "Windows":
    DEPENDENCY_LINKS.append("https://download.pytorch.org/whl/torch_stable.html")


def fetch_requirements(filename):
    with open(filename) as f:
        return [ln.strip() for ln in f.read().split("\n")]


setup(
    name="ipiqa",
    version="1.0.0",
    author="Bowen Qu",
    description="Code for IPIQA - BRINGING TEXTUAL PROMPT TO AI-GENERATED IMAGE QUALITY ASSESSMENT (ICME2024)",
    keywords="AI-Generated Images(AGIs), Image Quality Assessment (IQA), AI-Generated Image Quality Assessment (AGIQA), Multimodal Learning, Deep Learning, Library, PyTorch",
    license="3-Clause BSD",
    packages=find_namespace_packages(include="ipiqa.*"),
    install_requires=fetch_requirements("requirements.txt"),
    python_requires=">=3.8.0",
    include_package_data=True,
    dependency_links=DEPENDENCY_LINKS,
    zip_safe=False,
)