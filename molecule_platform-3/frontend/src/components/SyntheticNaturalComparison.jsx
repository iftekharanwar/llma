import React from 'react';
import {
  Box,
  VStack,
  Text,
  Icon,
  Grid,
  GridItem,
  Divider,
  HStack,
  Tooltip,
} from '@chakra-ui/react';
import { FiActivity, FiInfo } from 'react-icons/fi';

const ComparisonMetric = ({ label, synthetic, natural, info }) => (
  <Box
    p={3}
    borderRadius="md"
    transition="all 0.2s"
    _hover={{ bg: 'whiteAlpha.50' }}
  >
    <HStack spacing={2} mb={2}>
      <Text color="gray.300" fontSize="sm" fontWeight="medium">{label}</Text>
      {info && (
        <Tooltip label={info} hasArrow placement="top">
          <Box cursor="help">
            <Icon as={FiInfo} color="gray.400" boxSize={3} />
          </Box>
        </Tooltip>
      )}
    </HStack>
    <Grid templateColumns="1fr auto 1fr" gap={4} alignItems="center">
      <GridItem>
        <Text color="blue.300" fontSize="md" fontWeight="semibold">{synthetic}</Text>
      </GridItem>
      <GridItem>
        <Text color="gray.500" fontSize="xs" fontWeight="medium">vs</Text>
      </GridItem>
      <GridItem>
        <Text color="green.300" fontSize="md" fontWeight="semibold">{natural}</Text>
      </GridItem>
    </Grid>
  </Box>
);

const SyntheticNaturalComparison = ({ comparisonData }) => {
  if (!comparisonData) {
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
      >
        <VStack spacing={4} align="stretch">
          <HStack spacing={2}>
            <Icon as={FiActivity} color="blue.400" boxSize={5} />
            <Text fontSize="lg" fontWeight="medium" color="white">
              Synthetic vs Natural Comparison
            </Text>
          </HStack>
          <Box
            p={8}
            bg="whiteAlpha.50"
            borderRadius="lg"
            textAlign="center"
            border="1px dashed"
            borderColor="whiteAlpha.200"
          >
            <Icon as={FiActivity} boxSize={8} color="blue.300" mb={3} />
            <Text color="gray.300">
              Analyze a molecule to view comparison results
            </Text>
          </Box>
        </VStack>
      </Box>
    );
  }

  const { synthetic, natural } = comparisonData;

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
    >
      <VStack spacing={6} align="stretch">
        <HStack spacing={2}>
          <Icon as={FiActivity} color="blue.400" boxSize={5} />
          <Text fontSize="lg" fontWeight="medium" color="white">
            Synthetic vs Natural Comparison
          </Text>
        </HStack>

        <Grid templateColumns="repeat(2, 1fr)" gap={6}>
          {/* Synthetic Panel */}
          <GridItem
            bg="blue.900"
            p={5}
            borderRadius="lg"
            border="1px solid"
            borderColor="blue.700"
          >
            <Text color="blue.400" fontSize="md" fontWeight="semibold" mb={4}>
              Synthetic Molecule
            </Text>
            <VStack align="stretch" spacing={4}>
              <Box>
                <Text color="gray.400" fontSize="sm">Structure</Text>
                <Text color="white" fontSize="md">{synthetic.structure || 'N/A'}</Text>
              </Box>
              <Box>
                <Text color="gray.400" fontSize="sm">Properties</Text>
                <Text color="white" fontSize="md">
                  {synthetic.properties?.description || 'No properties available'}
                </Text>
              </Box>
            </VStack>
          </GridItem>

          {/* Natural Panel */}
          <GridItem
            bg="green.900"
            p={5}
            borderRadius="lg"
            border="1px solid"
            borderColor="green.700"
          >
            <Text color="green.400" fontSize="md" fontWeight="semibold" mb={4}>
              Natural Molecule
            </Text>
            <VStack align="stretch" spacing={4}>
              <Box>
                <Text color="gray.400" fontSize="sm">Structure</Text>
                <Text color="white" fontSize="md">{natural.structure || 'N/A'}</Text>
              </Box>
              <Box>
                <Text color="gray.400" fontSize="sm">Properties</Text>
                <Text color="white" fontSize="md">
                  {natural.properties?.description || 'No properties available'}
                </Text>
              </Box>
            </VStack>
          </GridItem>
        </Grid>

        <Divider borderColor="whiteAlpha.200" />

        <VStack spacing={4} align="stretch">
          {synthetic.properties && natural.properties && (
            <>
              <ComparisonMetric
                label="Molecular Weight"
                synthetic={synthetic.properties.molecular_weight}
                natural={natural.properties.molecular_weight}
                info="Molecular mass in g/mol"
              />
              <ComparisonMetric
                label="LogP"
                synthetic={synthetic.properties.logp}
                natural={natural.properties.logp}
                info="Lipophilicity measure"
              />
              <ComparisonMetric
                label="Risk Level"
                synthetic={synthetic.risks?.level}
                natural={natural.risks?.level}
                info="Overall risk assessment"
              />
              <ComparisonMetric
                label="Stability"
                synthetic={synthetic.stability?.level}
                natural={natural.stability?.level}
                info="Compound stability assessment"
              />
            </>
          )}
        </VStack>
      </VStack>
    </Box>
  );
};

export default SyntheticNaturalComparison;
