# dinkster-nodes

## Registry publisher credentials

First-party packs use the same self-service credentials as third-party packs.
Open [publisher accounts](https://id.dinkster.org/registry), sign in with the
Dinkster account or organization bound to the reserved `dinkster` publisher,
select that account or organization and choose **Refresh tokens**. The reserved
name is not available for a public claim; its initial audited binding is an
operator action, not a different token-minting route.

Choose an expiry at most 90 days away, optionally restrict the token to one pack,
then choose **Create token**. Save the one-time value in the publishing workflow's
secret store. Never commit it or print it in logs. To rotate, create a replacement
on this page, update the workflow secret, confirm a publication through the normal
workflow, then revoke the previous credential on the same page. Expired and
revoked tokens cannot publish. Publisher registration does not change admission,
scanning or the official publisher tier.
