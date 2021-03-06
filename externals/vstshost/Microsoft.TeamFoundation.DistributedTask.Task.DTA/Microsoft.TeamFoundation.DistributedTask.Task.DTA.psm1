#
# Module for 'Microsoft.TeamFoundation.DistributedTask.Task.DTA'
#

Import-Module '$PSScriptRoot\..\Microsoft.TeamFoundation.DistributedTask.Task.LegacySDK.dll'

Export-ModuleMember -Cmdlet @(
        'Invoke-DeployTestAgent',
        'Invoke-RunDistributedTests'
    )