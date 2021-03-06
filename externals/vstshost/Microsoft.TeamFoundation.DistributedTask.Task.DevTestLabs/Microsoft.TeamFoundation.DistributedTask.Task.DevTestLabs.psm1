#
# Module for 'Microsoft.TeamFoundation.DistributedTask.Task.DevTestLabs
#

Import-Module '$PSScriptRoot\..\Microsoft.TeamFoundation.DistributedTask.Task.LegacySDK.dll'

Export-ModuleMember -Cmdlet @(
        'Complete-EnvironmentOperation',
        'Complete-EnvironmentResourceOperation',
        'Complete-ResourceOperation',
        'Copy-FilesToAzureBlob',
        'Copy-FilesToRemote',
        'Copy-FilesToTargetMachine',
        'Copy-ToAzureMachines',
        'Get-Environment',
        'Get-EnvironmentProperty',
        'Get-EnvironmentResources',
        'Get-ExternalIpAddress',
        'Get-ParsedSessionVariables',
        'Get-ProviderData',
        'Invoke-BlockEnvironment',
        'Invoke-EnvironmentOperation',
        'Invoke-PsOnRemote',
        'Invoke-ResourceOperation',
        'Invoke-UnblockEnvironment'
        'New-OperationLog',
        'Register-Environment',
        'Register-EnvironmentDefinition',
        'Register-Provider',
        'Register-ProviderData',
        'Remove-Environment',
        'Remove-EnvironmentResources'
    )