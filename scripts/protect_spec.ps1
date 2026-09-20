# Apply local, file-scoped protection only after checking the recorded baseline.
# No unlock operation is provided. An owner/administrator can deliberately revoke
# this protection; Git does not transport these filesystem permissions.
$ErrorActionPreference = 'Stop'
$specRepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$specLockPath = Join-Path $specRepoRoot 'research\specification.lock.json'
$specRecord = Get-Content -LiteralPath $specLockPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($specRecord.schema -ne 'ergt-phi-specification-lock-v1' -or $specRecord.path -ne 'docs/MATHEMATICAL_SPEC.md') {
    throw 'Invalid mathematical specification lock.'
}
$specTarget = Join-Path $specRepoRoot 'docs\MATHEMATICAL_SPEC.md'
$specItem = Get-Item -LiteralPath $specTarget -Force
if (($specItem.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
    throw 'Refusing to protect a reparse point instead of the canonical file.'
}
$specDigest = (Get-FileHash -LiteralPath $specTarget -Algorithm SHA256).Hash.ToLowerInvariant()
if ($specItem.Length -ne $specRecord.bytes -or $specDigest -ne $specRecord.sha256) {
    throw 'Specification content differs from the baseline. Do not reseal changed content.'
}
$specAcl = Get-Acl -LiteralPath $specTarget
$specBackupDir = Join-Path $specRepoRoot '.cache\spec-protection'
[System.IO.Directory]::CreateDirectory($specBackupDir) | Out-Null
$specBackupPath = Join-Path $specBackupDir 'original-permissions.json'
if (-not (Test-Path -LiteralPath $specBackupPath)) {
    [ordered]@{
        path = $specRecord.path
        sha256 = $specDigest
        attributes = [int]$specItem.Attributes
        sddl = $specAcl.Sddl
    } | ConvertTo-Json | Set-Content -LiteralPath $specBackupPath -Encoding UTF8
}
# Set the attribute before denying attribute writes. Do not try to set it again
# on subsequent invocations, since the ACL correctly forbids that operation.
if (-not $specItem.IsReadOnly) { $specItem.IsReadOnly = $true }
$specEveryone = [System.Security.Principal.SecurityIdentifier]::new('S-1-1-0')
$specDenied = [System.Security.AccessControl.FileSystemRights]::Write -bor [System.Security.AccessControl.FileSystemRights]::Delete
$specRule = [System.Security.AccessControl.FileSystemAccessRule]::new(
    $specEveryone, $specDenied, [System.Security.AccessControl.AccessControlType]::Deny)
$specExisting = @($specAcl.Access | Where-Object {
    $_.IdentityReference.Translate([System.Security.Principal.SecurityIdentifier]).Value -eq 'S-1-1-0' -and
    $_.AccessControlType -eq [System.Security.AccessControl.AccessControlType]::Deny -and
    (($_.FileSystemRights -band $specDenied) -eq $specDenied) -and -not $_.IsInherited
})
if ($specExisting.Count -eq 0) {
    $specAcl.AddAccessRule($specRule) | Out-Null
    Set-Acl -LiteralPath $specTarget -AclObject $specAcl
}
# Windows can grant DELETE through the parent directory's DeleteChild right
# even when the file itself denies DELETE. Remove only that alternative route.
# This rule is NOT inherited; siblings keep their own ordinary Delete rights.
$specParent = Split-Path -Parent $specTarget
$specParentAcl = Get-Acl -LiteralPath $specParent
$specParentBackup = Join-Path $specBackupDir 'original-parent-permissions.json'
if (-not (Test-Path -LiteralPath $specParentBackup)) {
    [ordered]@{ path = 'docs'; sddl = $specParentAcl.Sddl } |
        ConvertTo-Json | Set-Content -LiteralPath $specParentBackup -Encoding UTF8
}
$specDeleteChild = [System.Security.AccessControl.FileSystemRights]::DeleteSubdirectoriesAndFiles
$specParentExisting = @($specParentAcl.Access | Where-Object {
    $_.IdentityReference.Translate([System.Security.Principal.SecurityIdentifier]).Value -eq 'S-1-1-0' -and
    $_.AccessControlType -eq [System.Security.AccessControl.AccessControlType]::Deny -and
    (($_.FileSystemRights -band $specDeleteChild) -eq $specDeleteChild) -and -not $_.IsInherited
})
if ($specParentExisting.Count -eq 0) {
    $specParentRule = [System.Security.AccessControl.FileSystemAccessRule]::new(
        $specEveryone, $specDeleteChild, [System.Security.AccessControl.AccessControlType]::Deny)
    $specParentAcl.AddAccessRule($specParentRule) | Out-Null
    Set-Acl -LiteralPath $specParent -AclObject $specParentAcl
}
$specAfter = Get-Item -LiteralPath $specTarget -Force
$specAfterDigest = (Get-FileHash -LiteralPath $specTarget -Algorithm SHA256).Hash.ToLowerInvariant()
if (-not $specAfter.IsReadOnly -or $specAfterDigest -ne $specDigest) { throw 'Protection verification failed.' }
$specWriteDenied = $false
try {
    # Open never truncates and no Write call is made, even if protection fails.
    $specHandle = [System.IO.File]::Open($specTarget, [System.IO.FileMode]::Open,
        [System.IO.FileAccess]::Write, [System.IO.FileShare]::ReadWrite)
    $specHandle.Dispose()
} catch [System.UnauthorizedAccessException] { $specWriteDenied = $true }
if (-not $specWriteDenied) { throw 'Write access was unexpectedly available.' }
[ordered]@{ pass = $true; path = $specRecord.path; sha256 = $specAfterDigest;
    read_only = $specAfter.IsReadOnly; write_open_denied = $specWriteDenied;
    protection = 'ReadOnly; file deny Write,Delete; non-inherited parent deny DeleteChild';
    local_only = $true } | ConvertTo-Json
