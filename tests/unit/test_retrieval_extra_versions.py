from importlib import metadata
from pathlib import Path
import tomllib

import pytest
from packaging.requirements import Requirement
from packaging.version import Version


REQUIRED_LOWER_BOUNDS = {
    "langchain-community": "0.3",
    "arxiv": "2",
    "pymupdf": "1.24",
    "exa-py": "2.12",
}


def _retrieval_requirements() -> dict[str, Requirement]:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    entries = project["project"]["optional-dependencies"]["retrieval"]
    return {Requirement(entry).name: Requirement(entry) for entry in entries}


def _declared_lower_bound(requirement: Requirement) -> Version:
    lower_bounds = [
        Version(spec.version)
        for spec in requirement.specifier
        if spec.operator in {">=", "=="}
    ]
    assert lower_bounds
    return max(lower_bounds)


def test_retrieval_extra_has_required_lower_bounds() -> None:
    requirements = _retrieval_requirements()

    for package_name, lower_bound in REQUIRED_LOWER_BOUNDS.items():
        requirement = requirements[package_name]
        assert _declared_lower_bound(requirement) >= Version(lower_bound)


@pytest.mark.parametrize("package_name", sorted(REQUIRED_LOWER_BOUNDS))
def test_installed_retrieval_packages_satisfy_lower_bounds(
    package_name: str,
) -> None:
    requirements = _retrieval_requirements()
    requirement = requirements[package_name]

    try:
        installed_version = Version(metadata.version(package_name))
    except metadata.PackageNotFoundError:
        pytest.skip(f"{package_name} is not installed in the base environment")

    assert installed_version >= _declared_lower_bound(requirement)
