import React from 'react';
import {
  Box,
  VStack,
  Text,
  Progress,
  Flex,
  Tooltip,
  Divider,
  List,
  ListItem,
  Badge,
  Icon,
  HStack,
} from '@chakra-ui/react';
import { FiAlertTriangle, FiInfo, FiShield } from 'react-icons/fi';

const RiskAssessment = ({ assessment, context }) => {
  if (!assessment) {
    return (
      <Box
        as="section"
        bg="whiteAlpha.50"
        borderRadius="xl"
        border="1px solid"
        borderColor="whiteAlpha.200"
        p={6}
      >
        <Text color="gray.300" fontSize="md" textAlign="center">
          Loading risk assessment data...
        </Text>
      </Box>
    );
  }

  const {
    toxicity_assessment,
    context_specific_risks,
    confidence_metrics,
    molecular_properties,
    overall_risk_level
  } = assessment;

  const getColorScheme = (level) => {
    switch (level?.toLowerCase()) {
      case 'low': return 'green';
      case 'medium': return 'yellow';
      case 'high': return 'red';
      default: return 'gray';
    }
  };

  return (
    <Box
      as="section"
      bg="whiteAlpha.50"
      borderRadius="xl"
      border="1px solid"
      borderColor="whiteAlpha.200"
      p={6}
      _hover={{ borderColor: 'whiteAlpha.300' }}
      transition="all 0.2s"
      height="100%"
    >
      <VStack spacing={4} align="stretch" height="100%">
        <HStack spacing={2} justify="space-between">
          <HStack>
            <Icon as={FiShield} color="blue.400" boxSize={5} />
            <Text fontSize="lg" fontWeight="semibold" color="white">
              Risk Assessment
            </Text>
          </HStack>
          <Badge
            colorScheme={getColorScheme(overall_risk_level)}
            px={3}
            py={1}
            borderRadius="full"
            fontSize="sm"
            fontWeight="medium"
          >
            {overall_risk_level}
          </Badge>
        </HStack>

        {/* Toxicity Assessment */}
        <Box>
          <Text fontSize="md" fontWeight="semibold" color="white" mb={3}>
            Toxicity Assessment
          </Text>
          <VStack spacing={3} align="stretch" bg="whiteAlpha.50" p={4} borderRadius="lg">
            {toxicity_assessment?.risks?.map((risk, index) => (
              <Box
                key={index}
                p={3}
                borderRadius="md"
                bg="whiteAlpha.100"
              >
                <Flex justify="space-between" mb={2} align="center">
                  <Text color="white" fontSize="sm" fontWeight="medium">
                    {risk.type}
                  </Text>
                  <Badge
                    colorScheme={getColorScheme(risk.level)}
                    px={2}
                    py={0.5}
                    borderRadius="full"
                  >
                    {risk.level}
                  </Badge>
                </Flex>
                <Text color="gray.300" fontSize="sm">
                  {risk.description}
                </Text>
                {risk.confidence && (
                  <Box mt={2}>
                    <Progress
                      value={risk.confidence}
                      size="xs"
                      colorScheme="blue"
                    />
                    <Text color="blue.300" fontSize="xs" mt={1}>
                      Confidence: {risk.confidence}%
                    </Text>
                  </Box>
                )}
              </Box>
            ))}
          </VStack>
        </Box>

        <Divider borderColor="whiteAlpha.200" />

        {/* Context-Specific Risks */}
        <Box>
          <Text fontSize="md" fontWeight="semibold" color="white" mb={3}>
            {context.charAt(0).toUpperCase() + context.slice(1)} Context Analysis
          </Text>
          <VStack spacing={3} align="stretch" bg="whiteAlpha.50" p={4} borderRadius="lg">
            {context_specific_risks?.risks?.map((risk, index) => (
              <Box
                key={index}
                p={3}
                borderRadius="md"
                bg="whiteAlpha.100"
              >
                <Text color="white" fontSize="sm" fontWeight="medium" mb={2}>
                  {risk}
                </Text>
                {context_specific_risks.recommendations?.[index] && (
                  <Text color="blue.300" fontSize="sm">
                    Recommendation: {context_specific_risks.recommendations[index]}
                  </Text>
                )}
              </Box>
            ))}
          </VStack>
        </Box>

        {/* Molecular Properties */}
        {molecular_properties && (
          <>
            <Divider borderColor="whiteAlpha.200" />
            <Box>
              <Text fontSize="md" fontWeight="semibold" color="white" mb={3}>
                Molecular Properties
              </Text>
              <List spacing={2} bg="whiteAlpha.50" p={4} borderRadius="lg">
                {Object.entries(molecular_properties).map(([key, value]) => (
                  <ListItem
                    key={key}
                    display="flex"
                    justifyContent="space-between"
                    alignItems="center"
                    p={2}
                  >
                    <Text color="gray.300" fontSize="sm">
                      {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </Text>
                    <Text color="blue.300" fontSize="sm" fontFamily="mono">
                      {typeof value === 'number' ? value.toFixed(2) : value}
                    </Text>
                  </ListItem>
                ))}
              </List>
            </Box>
          </>
        )}

        {/* Confidence Metrics */}
        <Box mt="auto">
          <HStack justify="space-between" mb={2}>
            <Text color="gray.300" fontSize="sm">
              Analysis Confidence
            </Text>
            <Tooltip
              label="Overall confidence in the risk assessment based on data quality and completeness"
              hasArrow
            >
              <Box cursor="help">
                <Icon as={FiInfo} color="gray.400" boxSize={4} />
              </Box>
            </Tooltip>
          </HStack>
          <Progress
            value={confidence_metrics?.prediction_reliability || 0}
            size="sm"
            colorScheme="blue"
            borderRadius="full"
          />
          <Text color="blue.300" fontSize="xs" mt={1}>
            {confidence_metrics?.prediction_reliability || 0}% Confidence
          </Text>
        </Box>
      </VStack>
    </Box>
  );
};

export default RiskAssessment;
