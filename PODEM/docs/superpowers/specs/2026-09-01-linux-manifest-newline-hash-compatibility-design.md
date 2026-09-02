# Linux Manifest Newline Hash Compatibility Design

## Problem

The SmartATPG training manifest was generated from a Windows checkout. Teacher
and profile JSON files therefore have SHA256 values for CRLF bytes. Git checks
the same tracked JSON files out with LF on Linux, so the Linux launcher rejects
semantically unchanged artifacts as modified.

## Design

Add one shared hash-validation helper to `rl_podem.artifact_paths`. Both the
Linux manifest relocator and curriculum trainer validation call this helper.
Validation first compares the exact file SHA256 with the manifest value. If
that fails and the artifact is JSON, it computes hashes for newline-normalized
LF and CRLF representations. The file is accepted only when one representation
exactly matches the recorded SHA256.

This is byte-preserving except for newline representation: JSON is not parsed,
reformatted, reordered, or otherwise canonicalized. Consequently, changes to
keys, values, whitespace, encoding, or content still fail unless the only byte
difference is LF versus CRLF. Circuit and fault-map files retain exact byte-hash
validation.

The portable manifest continues to reference the existing Linux checkout file;
the launcher does not rewrite source artifacts or alter the historical
manifest.

## Errors

An artifact that does not match either the exact hash or the permitted newline
variants raises the existing `artifact hash changed` error. Invalid or missing
paths continue to fail before training begins.

## Verification

Tests will verify that:

- a manifest hash generated from CRLF JSON accepts an LF checkout;
- exact hashes remain accepted;
- non-newline JSON content changes remain rejected;
- non-JSON artifacts do not receive newline normalization.
