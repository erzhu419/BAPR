# V19 audit-path amendment 1

The v19 GPU producers completed and synchronized all 36 policy bundles. Eight
of nine CPU audits then failed before rollout because the bundle validator read
`logs/protocol_signature.json` from the original GPU training directory. That
directory is deliberately not synchronized under the compact artifact policy;
the identical registered signature is already present and integrity-recorded
inside each specialist policy bundle.

Amendment 1 changes only this lookup location for replacement CPU audits. It
does not retrain a controller, change a parameter, change an event stream,
change an evaluation arm, or alter a decision threshold. The original v19
registration remains immutable. Replacement audits retain their original
output identities so the preregistered aggregate consumes the same nine cells.
