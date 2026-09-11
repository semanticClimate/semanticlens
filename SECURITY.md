# Security Policy

Security is important to SemanticLens particularly as the project processes scientific documents and may eventually integrate external models, services and data sources.

We appreciate responsible reports of potential security vulnerabilities.

## Supported Versions

SemanticLens is currently in active early stage development and has not yet published a stable release.

For now, security fixes are applied to the latest version of the `main` branch.

| Version                               | Supported         |
| ------------------------------------- | ----------------- |
| Latest `main` branch                  | ✅                 |
| Older commits or development branches | ❌                 |
| Tagged stable releases                | Not yet available |

This policy will be updated when SemanticLens begins publishing versioned releases.

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues, discussions, pull requests or other public channels.**

If the repository provides a **Report a vulnerability** option under GitHub's Security section, please use it to submit the report privately.

If private vulnerability reporting is not available, open a public issue titled:

`Security contact request`

Do **not** include any vulnerability details in that issue. A maintainer can then establish an appropriate private communication channel with you.

A useful vulnerability report should include, where applicable:

* A clear description of the vulnerability
* The affected component or functionality
* Steps required to reproduce the issue
* The potential security impact
* Relevant environment or configuration information
* Proof-of-concept details, logs, or screenshots when necessary
* Any suggested mitigation or fix, if known

Please remove unrelated credentials, personal information, private datasets, API keys, tokens or other sensitive information from reports.

## What to Report

Examples of issues that should be reported privately include:

* Exposure of credentials, tokens, secrets or sensitive information
* Unauthorized access to data or functionality
* Arbitrary code execution
* Unsafe file or document processing
* Path traversal or unintended file access
* Security issues involving external integrations or dependencies
* Authentication or authorization vulnerabilities, if such functionality is introduced
* Vulnerabilities that could compromise users, systems or project infrastructure

Normal bugs that do not have security implications should be reported through the regular GitHub issue tracker.

## Responsible Disclosure

We ask security researchers and contributors to:

* Report vulnerabilities privately before public disclosure
* Avoid exploiting a vulnerability beyond what is necessary to demonstrate the issue
* Avoid accessing, modifying or deleting data that does not belong to you
* Avoid actions that could disrupt project infrastructure or services
* Allow maintainers a reasonable opportunity to investigate and address the issue before public disclosure

Maintainers will work with reporters in good faith to understand and resolve legitimate security concerns.

## Disclosure and Fixes

When a vulnerability is confirmed, maintainers will aim to:

1. Assess the impact and affected components
2. Develop and review an appropriate fix
3. Coordinate disclosure with the reporter when appropriate
4. Publish relevant security information once a fix or mitigation is available

Specific response or resolution times are not guaranteed while SemanticLens is in early development.

## Scope

This policy applies to security vulnerabilities in code and resources maintained directly within the SemanticLens project.

Security issues affecting third-party software or services should generally be reported to the corresponding upstream project or provider. If a third party vulnerability creates a specific security risk for SemanticLens users, it may also be reported to the SemanticLens maintainers.

## Safe Harbor

Good faith security research performed in accordance with this policy is welcome.

Please act responsibly, avoid privacy violations or service disruption and limit testing to what is necessary to demonstrate a potential vulnerability.
