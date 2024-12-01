import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  useColorModeValue,
  Text,
  Heading,
  Flex,
  Icon,
} from '@chakra-ui/react';
import { FiActivity } from 'react-icons/fi';
import MoleculeInput from './components/MoleculeInput';
import MoleculeViewer3D from './components/MoleculeViewer3D';
import SimilaritySearchResults from './components/SimilaritySearchResults';
import RiskAssessment from './components/RiskAssessment';
import MedicalRecordsDashboard from './components/MedicalRecordsDashboard';
import SynthMolAssistant from './components/SynthMolAssistant';
import SyntheticNaturalComparison from './components/SyntheticNaturalComparison';
import AnalysisPanel from './components/AnalysisPanel';
import MoleculeViewerTest from './test/MoleculeViewerTest';
import MoleculeFormatTester from './test/MoleculeFormatTester';
import Visualization3DTest from './test/Visualization3DTest';

function App() {
  const [moleculeData, setMoleculeData] = useState(null);
  const [isTestMode, setIsTestMode] = useState(false);
  const [testComponent, setTestComponent] = useState('viewer'); // Add state for test component selection

  const handleAnalysis = (data) => {
    setMoleculeData(data);
  };

  const handleChatQuestion = async (question, currentMoleculeData) => {
    // Will be implemented during backend integration
    return "Please wait while I analyze your question about the molecule...";
  };

  // Development test mode toggle
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    setIsTestMode(params.get('test') === 'true');
    setTestComponent(params.get('component') || 'viewer');
  }, []);

  if (isTestMode) {
    return (
      <Box bg="#0D121E" minH="100vh" color="gray.100">
        {testComponent === 'format' ? (
          <MoleculeFormatTester />
        ) : testComponent === '3d' ? (
          <Visualization3DTest />
        ) : (
          <MoleculeViewerTest />
        )}
      </Box>
    );
  }

  return (
    <Box bg="#0D121E" minH="100vh" color="gray.100">
      <Flex direction="column" maxW="1800px" mx="auto" p={[6, 8, 10]}>
        <Flex align="center" mb={10}>
          <Icon as={FiActivity} color="blue.400" boxSize={7} mr={3} />
          <Heading size="lg" color="white" letterSpacing="tight">
            Synthetic Molecule Analysis Platform
          </Heading>
        </Flex>

        <Grid
          templateColumns={["1fr", "1fr", "repeat(12, 1fr)"]}
          gap={8}
          mb={8}
        >
          {/* Left Column - Input and 3D View */}
          <Box gridColumn={["span 12", "span 12", "span 7"]} display="flex" flexDirection="column" gap={8}>
            <Box
              bg="rgba(255, 255, 255, 0.03)"
              p={8}
              borderRadius="2xl"
              border="1px solid"
              borderColor="whiteAlpha.100"
              _hover={{ borderColor: 'blue.400', boxShadow: '0 8px 16px rgba(0, 0, 0, 0.2)' }}
              transition="all 0.3s"
            >
              <MoleculeInput onAnalyze={handleAnalysis} />
            </Box>

            <Box
              bg="rgba(255, 255, 255, 0.03)"
              p={8}
              borderRadius="2xl"
              border="1px solid"
              borderColor="whiteAlpha.100"
              _hover={{ borderColor: 'blue.400', boxShadow: '0 8px 16px rgba(0, 0, 0, 0.2)' }}
              transition="all 0.3s"
              flex="1"
              minH="600px"
              position="relative"
              overflow="hidden"
            >
              <Heading size="md" mb={6} color="blue.200" letterSpacing="wide">3D Molecular Structure</Heading>
              {moleculeData ? (
                <Box position="relative" height="calc(100% - 48px)">
                  <MoleculeViewer3D
                    moleculeData={moleculeData.structure_3d || moleculeData.structure}
                    format={moleculeData.format || 'mol'}
                    similarMolecules={moleculeData.similar_molecules || []}
                    onError={(error) => {
                      console.error('Molecule viewer error:', error);
                    }}
                  />
                </Box>
              ) : (
                <Flex h="100%" align="center" justify="center">
                  <Text color="whiteAlpha.600" fontSize="lg">
                    Upload or input a molecular structure to view its 3D visualization
                  </Text>
                </Flex>
              )}
            </Box>
          </Box>

          {/* Right Column - Analysis Results */}
          <Box gridColumn={["span 12", "span 12", "span 5"]} display="flex" flexDirection="column" gap={6}>
            {/* Similarity Search Results */}
            <AnalysisPanel title="Similarity Search Results">
              <SimilaritySearchResults results={moleculeData?.similar_molecules} />
            </AnalysisPanel>

            {/* Risk Assessment */}
            <AnalysisPanel title="Risk Assessment">
              <RiskAssessment riskData={moleculeData?.risk_assessment} />
            </AnalysisPanel>

            {/* Synthetic vs Natural Comparison */}
            <AnalysisPanel title="Synthetic vs Natural Comparison">
              <SyntheticNaturalComparison comparisonData={moleculeData?.comparison_data} />
            </AnalysisPanel>

            {/* Medical Records Dashboard */}
            <AnalysisPanel title="Medical Records Dashboard">
              <MedicalRecordsDashboard records={moleculeData?.medical_records} />
            </AnalysisPanel>

            {/* SynthMol Assistant */}
            <AnalysisPanel title="SynthMol Assistant">
              <SynthMolAssistant
                moleculeData={moleculeData}
                onAskQuestion={handleChatQuestion}
                isLoading={!moleculeData}
              />
            </AnalysisPanel>
          </Box>
        </Grid>
      </Flex>
    </Box>
  );
}

export default App;
