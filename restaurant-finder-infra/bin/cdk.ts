#!/usr/bin/env node
import * as cdk from "aws-cdk-lib";
import { BaseStackProps } from "../lib/types";
import { EcrStack, AgentCoreStack } from "../lib/stacks";

const app = new cdk.App();

// Override image URI via context variable:
//   npx cdk deploy --all -c imageUri=<account>.dkr.ecr.<region>.amazonaws.com/restaurantfinder-agent:<tag>
const existingImageUri = app.node.tryGetContext("imageUri") as
  | string
  | undefined;

const deploymentProps: BaseStackProps = {
  appName: "restaurantFinder",
  /* If you don't specify 'env', this stack will be environment-agnostic.
   * Account/Region-dependent features and context lookups will not work,
   * but a single synthesized template can be deployed anywhere. */

  /* Uncomment the next line to specialize this stack for the AWS Account
   * and Region that are implied by the current CLI configuration. */
  // env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: process.env.CDK_DEFAULT_REGION },

  /* Uncomment the next line if you know exactly what Account and Region you
   * want to deploy the stack to. */
  // env: { account: '123456789012', region: 'us-east-1' },

  /* For more information, see https://docs.aws.amazon.com/cdk/latest/guide/environments.html */
};

// ECR repository is always created as infrastructure
const ecrStack = new EcrStack(
  app,
  `restaurantFinder-EcrStack`,
  deploymentProps,
);

// Determine image URI:
// - If provided via context (-c imageUri=...), use that specific image
// - Otherwise, default to the ECR repo with :latest tag
const imageUri = existingImageUri || `${ecrStack.repositoryUri}:latest`;

if (existingImageUri) {
  console.log(`Using provided image URI: ${existingImageUri}`);
} else {
  console.log(`Using default ECR image URI: ${imageUri}`);
}

const agentCoreStack = new AgentCoreStack(
  app,
  `restaurantFinder-AgentCoreStack`,
  {
    ...deploymentProps,
    imageUri: imageUri,
  },
);

agentCoreStack.addDependency(ecrStack);
