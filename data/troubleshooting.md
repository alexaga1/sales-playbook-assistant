# Northwind Supply Co. — Troubleshooting

## Cannot Log In

First confirm the user is on the correct portal URL and not a bookmarked staging link. Password resets go to the email on file and expire after 60 minutes. If single sign-on is enabled, login issues are usually an identity-provider problem on the customer's side, not a Northwind problem.

## Orders Not Syncing

If orders placed in the portal do not appear in the customer's procurement system, check the integration status page first. A red status on the Salesforce or QuickBooks connector usually means an expired API token, which an admin can refresh in Settings, then Integrations. Sync runs every 15 minutes, so allow one full cycle before retesting.

## Reports Running Slowly

Reports over a full year of data can take up to two minutes. Narrowing the date range to a single quarter usually brings load time under 15 seconds. If a report still times out, it is often because a department filter was left blank, which pulls every cost center at once.
