import * as cdk from "aws-cdk-lib/core";
import { Construct } from "constructs/lib/construct";
import * as bedrockagentcore from "aws-cdk-lib/aws-bedrockagentcore";
import * as iam from "aws-cdk-lib/aws-iam";
import * as lambda from "aws-cdk-lib/aws-lambda";
import * as logs from "aws-cdk-lib/aws-logs";
import * as secretsmanager from "aws-cdk-lib/aws-secretsmanager";
import { BaseStackProps } from "../types";
import * as path from "path";
import * as fs from "fs";

export interface AgentCoreStackProps extends BaseStackProps {
  imageUri: string;
}

export class AgentCoreStack extends cdk.Stack {
  readonly agentCoreRuntime: bedrockagentcore.CfnRuntime;
  readonly agentCoreGateway: bedrockagentcore.CfnGateway;
  readonly agentCoreMemory: bedrockagentcore.CfnMemory;
  readonly mcpLambda: lambda.Function;

  constructor(scope: Construct, id: string, props: AgentCoreStackProps) {
    super(scope, id, props);

    const region = cdk.Stack.of(this).region;
    const accountId = cdk.Stack.of(this).account;

    /*****************************
     * AgentCore Gateway
     ******************************/

    // Secret for the restaurant search API key used by the MCP Lambda.
    // After deployment, update with your key:
    //   aws secretsmanager put-secret-value --secret-id <appName>/restaurant-search-key \
    //     --secret-string '{"api_key":"your-actual-key"}'
    const searchSecretName = `${props.appName}/restaurant-search-key`;
    const searchSecret = new secretsmanager.Secret(
      this,
      `${props.appName}-SearchSecret`,
      {
        secretName: searchSecretName,
        description:
          "API key for restaurant search. Update with your key after deployment.",
        secretObjectValue: {
          api_key: cdk.SecretValue.unsafePlainText(""),
        },
      },
    );

    this.mcpLambda = new lambda.Function(this, `${props.appName}-McpLambda`, {
      runtime: lambda.Runtime.PYTHON_3_12,
      handler: "handler.lambda_handler",
      code: lambda.AssetCode.fromAsset(
        path.join(__dirname, "../../mcp/lambda"),
      ),
      memorySize: 256,
      environment: {
        SEARCH_SECRET_NAME: searchSecretName,
      },
      timeout: cdk.Duration.seconds(90),
    });

    searchSecret.grantRead(this.mcpLambda);

    const agentCoreGatewayRole = new iam.Role(
      this,
      `${props.appName}-AgentCoreGatewayRole`,
      {
        assumedBy: new iam.ServicePrincipal("bedrock-agentcore.amazonaws.com"),
        description: "IAM role for Bedrock AgentCore Runtime",
      },
    );

    const lambdaInvokeGrant = this.mcpLambda.grantInvoke(agentCoreGatewayRole);

    // Add explicit Lambda resource-based policy for BedrockAgentCore service
    const lambdaPermission = new lambda.CfnPermission(
      this,
      `${props.appName}-McpLambdaPermission`,
      {
        action: "lambda:InvokeFunction",
        functionName: this.mcpLambda.functionName,
        principal: "bedrock-agentcore.amazonaws.com",
        sourceArn: `arn:aws:bedrock-agentcore:${region}:${accountId}:gateway/*`,
      },
    );

    // Create gateway resource
    // ⚠️ WARNING: Authorization is disabled (authorizerType: "NONE")
    // This configuration is NOT RECOMMENDED for production environments.
    // We are using this setup for demo/testing purposes only.
    // For production, use one of the following authorizer types:
    //   - "CUSTOM_JWT" with Cognito or another OIDC provider
    //   - "IAM" for AWS IAM-based authentication
    // See: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway-authorization.html
    this.agentCoreGateway = new bedrockagentcore.CfnGateway(
      this,
      `${props.appName}-AgentCoreGateway`,
      {
        name: `${props.appName}-Gateway`,
        protocolType: "MCP",
        roleArn: agentCoreGatewayRole.roleArn,
        authorizerType: "NONE",
      },
    );

    // Load tool schema from JSON file (co-located with Lambda code for easier maintenance)
    const toolsSchemaPath = path.join(
      __dirname,
      "../../mcp/lambda/tools_schema.json",
    );
    const toolsSchema = JSON.parse(fs.readFileSync(toolsSchemaPath, "utf-8"));

    const gatewayTarget = new bedrockagentcore.CfnGatewayTarget(
      this,
      `${props.appName}-AgentCoreGatewayLambdaTarget`,
      {
        name: `${props.appName}-Target`,
        gatewayIdentifier: this.agentCoreGateway.attrGatewayIdentifier,
        credentialProviderConfigurations: [
          {
            credentialProviderType: "GATEWAY_IAM_ROLE",
          },
        ],
        targetConfiguration: {
          mcp: {
            lambda: {
              lambdaArn: this.mcpLambda.functionArn,
              toolSchema: {
                inlinePayload: toolsSchema,
              },
            },
          },
        },
      },
    );

    // Ensure Lambda permission is created before the GatewayTarget
    gatewayTarget.addDependency(lambdaPermission);

    // Ensure the IAM policy granting the Gateway role permission to invoke Lambda
    // is fully created before the GatewayTarget. This prevents the race condition
    // where BedrockAgentCore validates the role permissions before IAM propagates.
    const roleDefaultPolicy = agentCoreGatewayRole.node.tryFindChild(
      "DefaultPolicy",
    ) as iam.Policy | undefined;
    if (roleDefaultPolicy) {
      gatewayTarget.node.addDependency(roleDefaultPolicy);
    }

    void lambdaInvokeGrant;

    /*****************************
     * AgentCore Memory
     ******************************/

    this.agentCoreMemory = new bedrockagentcore.CfnMemory(
      this,
      `${props.appName}-AgentCoreMemory`,
      {
        name: `${props.appName}_Memory`,
        eventExpiryDuration: 30,
        description: `${props.appName} Memory resource with 30 days event expiry`,
        // Memory strategies matching the Python infrastructure/memory.py configuration
        memoryStrategies: [
          {
            // UserPreferenceStrategy - extracts and stores user preferences
            userPreferenceMemoryStrategy: {
              name: "user_preference_strategy",
              namespaces: ["/users/{actorId}/preferences"],
            },
          },
          {
            // SemanticStrategy - extracts semantic facts from conversations
            semanticMemoryStrategy: {
              name: "semantic_strategy",
              namespaces: ["/conversations/{actorId}/facts"],
            },
          },
          {
            // SummaryStrategy - generates conversation summaries
            summaryMemoryStrategy: {
              name: "summary_strategy",
              namespaces: ["/conversations/{sessionId}/summaries"],
            },
          },
        ],
      },
    );

    /*****************************
     * AgentCore Runtime
     ******************************/

    // taken from https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-permissions.html#runtime-permissions-execution
    const runtimePolicy = new iam.PolicyDocument({
      statements: [
        new iam.PolicyStatement({
          sid: "ECRImageAccess",
          effect: iam.Effect.ALLOW,
          actions: ["ecr:BatchGetImage", "ecr:GetDownloadUrlForLayer"],
          resources: [`arn:aws:ecr:${region}:${accountId}:repository/*`],
        }),
        new iam.PolicyStatement({
          effect: iam.Effect.ALLOW,
          actions: ["logs:DescribeLogStreams", "logs:CreateLogGroup"],
          resources: [
            `arn:aws:logs:${region}:${accountId}:log-group:/aws/bedrock-agentcore/runtimes/*`,
          ],
        }),
        new iam.PolicyStatement({
          effect: iam.Effect.ALLOW,
          actions: ["logs:DescribeLogGroups"],
          resources: [`arn:aws:logs:${region}:${accountId}:log-group:*`],
        }),
        new iam.PolicyStatement({
          effect: iam.Effect.ALLOW,
          actions: ["logs:CreateLogStream", "logs:PutLogEvents"],
          resources: [
            `arn:aws:logs:${region}:${accountId}:log-group:/aws/bedrock-agentcore/runtimes/*:log-stream:*`,
          ],
        }),
        new iam.PolicyStatement({
          sid: "ECRTokenAccess",
          effect: iam.Effect.ALLOW,
          actions: ["ecr:GetAuthorizationToken"],
          resources: ["*"],
        }),
        new iam.PolicyStatement({
          effect: iam.Effect.ALLOW,
          actions: [
            "xray:PutTraceSegments",
            "xray:PutTelemetryRecords",
            "xray:GetSamplingRules",
            "xray:GetSamplingTargets",
          ],
          resources: ["*"],
        }),
        new iam.PolicyStatement({
          effect: iam.Effect.ALLOW,
          actions: ["cloudwatch:PutMetricData"],
          resources: ["*"],
          conditions: {
            StringEquals: { "cloudwatch:namespace": "bedrock-agentcore" },
          },
        }),
        // OpenTelemetry OTLP exporter permissions for CloudWatch observability
        new iam.PolicyStatement({
          sid: "OTLPCloudWatchExport",
          effect: iam.Effect.ALLOW,
          actions: [
            "logs:PutLogEvents",
            "logs:CreateLogStream",
            "logs:CreateLogGroup",
            "logs:DescribeLogStreams",
          ],
          resources: [
            `arn:aws:logs:${region}:${accountId}:log-group:/aws/vendedlogs/bedrock-agentcore/*`,
            `arn:aws:logs:${region}:${accountId}:log-group:/aws/vendedlogs/bedrock-agentcore/*:log-stream:*`,
            `arn:aws:logs:${region}:${accountId}:log-group:aws/spans:*`,
          ],
        }),
        new iam.PolicyStatement({
          sid: "GetAgentAccessToken",
          effect: iam.Effect.ALLOW,
          actions: [
            "bedrock-agentcore:GetWorkloadAccessToken",
            "bedrock-agentcore:GetWorkloadAccessTokenForJWT",
            "bedrock-agentcore:GetWorkloadAccessTokenForUserId",
          ],
          resources: [
            `arn:aws:bedrock-agentcore:${region}:${accountId}:workload-identity-directory/default`,
            `arn:aws:bedrock-agentcore:${region}:${accountId}:workload-identity-directory/default/workload-identity/agentName-*`,
          ],
        }),
        new iam.PolicyStatement({
          sid: "BedrockModelInvocation",
          effect: iam.Effect.ALLOW,
          actions: [
            "bedrock:InvokeModel",
            "bedrock:InvokeModelWithResponseStream",
          ],
          resources: [
            `arn:aws:bedrock:*::foundation-model/*`,
            `arn:aws:bedrock:${region}:${accountId}:*`,
          ],
        }),
        new iam.PolicyStatement({
          sid: "BedrockPromptsAccess",
          effect: iam.Effect.ALLOW,
          actions: ["bedrock:ListPrompts", "bedrock:GetPrompt"],
          resources: [`arn:aws:bedrock:${region}:${accountId}:prompt/*`],
        }),
        // Guardrails - create, list, version, and apply
        new iam.PolicyStatement({
          sid: "BedrockGuardrailsManagement",
          effect: iam.Effect.ALLOW,
          actions: [
            "bedrock:CreateGuardrail",
            "bedrock:ListGuardrails",
            "bedrock:GetGuardrail",
            "bedrock:CreateGuardrailVersion",
            "bedrock:UpdateGuardrail",
          ],
          resources: [`arn:aws:bedrock:${region}:${accountId}:guardrail/*`],
        }),
        new iam.PolicyStatement({
          sid: "BedrockGuardrailsList",
          effect: iam.Effect.ALLOW,
          actions: ["bedrock:ListGuardrails"],
          resources: ["*"],
        }),
        // Bedrock Runtime - ApplyGuardrail and Converse API
        new iam.PolicyStatement({
          sid: "BedrockRuntimeOperations",
          effect: iam.Effect.ALLOW,
          actions: [
            "bedrock:ApplyGuardrail",
            "bedrock:Converse",
            "bedrock:ConverseStream",
          ],
          resources: [
            `arn:aws:bedrock:${region}:${accountId}:guardrail/*`,
            `arn:aws:bedrock:*::foundation-model/*`,
          ],
        }),
        // AgentCore Memory operations - full access to memory resources
        new iam.PolicyStatement({
          sid: "BedrockAgentCoreMemory",
          effect: iam.Effect.ALLOW,
          actions: ["bedrock-agentcore:*"],
          resources: [
            `arn:aws:bedrock-agentcore:${region}:${accountId}:memory/*`,
          ],
        }),
        // AgentCore Browser operations (AWS-managed browser resource)
        new iam.PolicyStatement({
          sid: "BedrockAgentCoreBrowser",
          effect: iam.Effect.ALLOW,
          actions: [
            "bedrock-agentcore:StartBrowserSession",
            "bedrock-agentcore:StopBrowserSession",
            "bedrock-agentcore:GetBrowserSession",
            "bedrock-agentcore:SendBrowserCommand",
          ],
          resources: [`arn:aws:bedrock-agentcore:${region}:aws:browser/*`],
        }),
        // CloudWatch Logs Delivery API for application observability
        new iam.PolicyStatement({
          sid: "CloudWatchLogsDelivery",
          effect: iam.Effect.ALLOW,
          actions: [
            "logs:PutDeliverySource",
            "logs:PutDeliveryDestination",
            "logs:CreateDelivery",
            "logs:GetDeliverySource",
            "logs:GetDeliveryDestination",
            "logs:GetDelivery",
            "logs:DeleteDeliverySource",
            "logs:DeleteDeliveryDestination",
            "logs:DeleteDelivery",
          ],
          resources: ["*"],
        }),
        // S3 Vector Store operations
        new iam.PolicyStatement({
          sid: "S3VectorStoreAccess",
          effect: iam.Effect.ALLOW,
          actions: [
            "s3vectors:QueryVectors",
            "s3vectors:PutVectors",
            "s3vectors:GetVectors",
            "s3vectors:DeleteVectors",
          ],
          resources: [`arn:aws:s3vectors:${region}:${accountId}:bucket/*`],
        }),
        // S3 bucket access for documents and vectors
        new iam.PolicyStatement({
          sid: "S3BucketAccess",
          effect: iam.Effect.ALLOW,
          actions: [
            "s3:GetObject",
            "s3:PutObject",
            "s3:DeleteObject",
            "s3:ListBucket",
          ],
          resources: [
            `arn:aws:s3:::${props.appName.toLowerCase()}-*`,
            `arn:aws:s3:::${props.appName.toLowerCase()}-*/*`,
          ],
        }),
      ],
    });

    const runtimeRole = new iam.Role(
      this,
      `${props.appName}-AgentCoreRuntimeRole`,
      {
        assumedBy: new iam.ServicePrincipal("bedrock-agentcore.amazonaws.com"),
        description: "IAM role for Bedrock AgentCore Runtime",
        inlinePolicies: {
          RuntimeAccessPolicy: runtimePolicy,
        },
      },
    );

    this.agentCoreRuntime = new bedrockagentcore.CfnRuntime(
      this,
      `${props.appName}-AgentCoreRuntime`,
      {
        agentRuntimeArtifact: {
          containerConfiguration: {
            containerUri: props.imageUri,
          },
        },
        agentRuntimeName: `${props.appName}_Agent`,
        protocolConfiguration: "HTTP",
        networkConfiguration: {
          networkMode: "PUBLIC",
        },
        roleArn: runtimeRole.roleArn,
        environmentVariables: {
          // AWS Configuration
          AWS_REGION: region,

          // AgentCore Resources
          GATEWAY_URL: this.agentCoreGateway.attrGatewayUrl,
          GATEWAY_ID: this.agentCoreGateway.attrGatewayIdentifier,
          MEMORY_ID: this.agentCoreMemory.attrMemoryId,

          // Feature Flags
          ENABLE_BROWSER_TOOLS: "true",
          GUARDRAIL_ENABLED: "true",

          // OpenTelemetry Observability Configuration
          // These enable CloudWatch GenAI Observability integration
          AGENT_OBSERVABILITY_ENABLED: "true",
          OTEL_SERVICE_NAME: `${props.appName}-agent`,
          OTEL_PYTHON_DISTRO: "aws_distro",
          OTEL_PYTHON_CONFIGURATOR: "aws_configurator",
          OTEL_EXPORTER_OTLP_PROTOCOL: "http/protobuf",
          OTEL_TRACES_EXPORTER: "otlp",
          OTEL_METRICS_EXPORTER: "otlp",
          OTEL_LOGS_EXPORTER: "otlp",

          // OTLP endpoint for CloudWatch (region-specific)
          OTEL_EXPORTER_OTLP_ENDPOINT: `https://xray.${region}.amazonaws.com`,

          // Trace propagation format
          OTEL_PROPAGATORS: "xray,tracecontext,baggage",

          // CloudWatch log configuration for OTLP
          OTEL_RESOURCE_ATTRIBUTES: `service.name=${props.appName}-agent,aws.log.group.names=/aws/bedrock-agentcore/runtimes/${props.appName}-agent,cloud.region=${region}`,
          OTEL_EXPORTER_OTLP_LOGS_HEADERS: `x-aws-log-group=/aws/bedrock-agentcore/runtimes/${props.appName}-agent,x-aws-log-stream=runtime-logs,x-aws-metric-namespace=bedrock-agentcore`,
        },
      },
    );

    // DEFAULT endpoint is automatically created by AgentCore and always points to the latest published version
    // No explicit endpoint creation needed - use the DEFAULT endpoint for always-latest behavior
    // https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agent-runtime-versioning.html

    /*****************************
     * Transaction Search Config
     ******************************/

    // Resource policy to allow X-Ray to write spans to CloudWatch Logs
    new logs.CfnResourcePolicy(this, `${props.appName}-XRayResourcePolicy`, {
      policyName: `${props.appName}-XRayCloudWatchLogsAccess`,
      policyDocument: JSON.stringify({
        Version: "2012-10-17",
        Statement: [
          {
            Effect: "Allow",
            Principal: {
              Service: "xray.amazonaws.com",
            },
            Action: ["logs:PutLogEvents", "logs:CreateLogStream"],
            Resource: `arn:aws:logs:${region}:${accountId}:log-group:aws/spans:*`,
          },
        ],
      }),
    });

    // =============================================================================
    // Stack Outputs - Important IDs and URLs for reference
    // =============================================================================

    // Gateway outputs
    new cdk.CfnOutput(this, "GatewayUrl", {
      value: this.agentCoreGateway.attrGatewayUrl,
      description: "AgentCore Gateway URL for MCP connections",
      exportName: `${props.appName}-GatewayUrl`,
    });

    new cdk.CfnOutput(this, "GatewayId", {
      value: this.agentCoreGateway.attrGatewayIdentifier,
      description: "AgentCore Gateway Identifier",
      exportName: `${props.appName}-GatewayId`,
    });

    // Memory outputs
    new cdk.CfnOutput(this, "MemoryId", {
      value: this.agentCoreMemory.attrMemoryId,
      description: "AgentCore Memory ID",
      exportName: `${props.appName}-MemoryId`,
    });

    // Runtime outputs
    new cdk.CfnOutput(this, "RuntimeId", {
      value: this.agentCoreRuntime.attrAgentRuntimeId,
      description: "AgentCore Runtime ID",
      exportName: `${props.appName}-RuntimeId`,
    });

    new cdk.CfnOutput(this, "RuntimeArn", {
      value: this.agentCoreRuntime.attrAgentRuntimeArn,
      description: "AgentCore Runtime ARN",
      exportName: `${props.appName}-RuntimeArn`,
    });

    // Lambda outputs
    new cdk.CfnOutput(this, "McpLambdaArn", {
      value: this.mcpLambda.functionArn,
      description: "MCP Lambda Function ARN",
      exportName: `${props.appName}-McpLambdaArn`,
    });

    new cdk.CfnOutput(this, "McpLambdaName", {
      value: this.mcpLambda.functionName,
      description: "MCP Lambda Function Name",
      exportName: `${props.appName}-McpLambdaName`,
    });

    // Additional Gateway outputs
    new cdk.CfnOutput(this, "GatewayArn", {
      value: this.agentCoreGateway.attrGatewayArn,
      description: "AgentCore Gateway ARN",
      exportName: `${props.appName}-GatewayArn`,
    });

    new cdk.CfnOutput(this, "SearchSecretArn", {
      value: searchSecret.secretArn,
      description: "Search API secret ARN",
      exportName: `${props.appName}-SearchSecretArn`,
    });

    // Additional Memory outputs
    new cdk.CfnOutput(this, "MemoryArn", {
      value: this.agentCoreMemory.attrMemoryArn,
      description: "AgentCore Memory ARN",
      exportName: `${props.appName}-MemoryArn`,
    });

    // Runtime role ARN (needed by deploy-image workflow to call update-agent-runtime)
    new cdk.CfnOutput(this, "RuntimeRoleArn", {
      value: runtimeRole.roleArn,
      description:
        "AgentCore Runtime Role ARN (needed for update-agent-runtime)",
      exportName: `${props.appName}-RuntimeRoleArn`,
    });
  }
}
