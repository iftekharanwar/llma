import React, { useEffect, useRef, useState } from 'react';
import { Box, Text, Spinner, Center } from '@chakra-ui/react';
import * as $3Dmol from '3dmol';

const MoleculeViewer = ({ structure, format, properties }) => {
  const viewerRef = useRef(null);
  const containerRef = useRef(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!structure || !format || !containerRef.current) return;

    setIsLoading(true);
    setError(null);

    try {
      // Clear previous viewer if it exists
      if (viewerRef.current) {
        viewerRef.current.clear();
      }

      // Initialize 3Dmol viewer
      const viewer = $3Dmol.createViewer(containerRef.current, {
        backgroundColor: 'white',
      });
      viewerRef.current = viewer;

      // Handle different input formats
      switch (format) {
        case 'smiles':
          viewer.addModel(structure, 'mol');
          viewer.setStyle({}, { stick: {} });
          viewer.zoomTo();
          break;

        case 'mol':
          viewer.addModel(structure, 'mol');
          viewer.setStyle({}, { stick: {} });
          viewer.zoomTo();
          break;

        case 'pdb':
          viewer.addModel(structure, 'pdb');
          viewer.setStyle({}, {
            cartoon: { color: 'spectrum' },
            stick: { radius: 0.15 }
          });
          viewer.zoomTo();
          break;

        default:
          throw new Error(`Unsupported format: ${format}`);
      }

      // Add labels for atoms
      if (format !== 'pdb') {
        viewer.addLabels({
          font: '12px Arial',
          alignment: 'center'
        });
      }

      // Enable rotation and zoom controls
      viewer.rotate(90);
      viewer.render();
      viewer.zoom(0.8);

      setIsLoading(false);
    } catch (err) {
      console.error('Error rendering molecule:', err);
      setError(err.message);
      setIsLoading(false);
    }

    return () => {
      if (viewerRef.current) {
        viewerRef.current.clear();
      }
    };
  }, [structure, format]);

  if (error) {
    return (
      <Center h="100%" bg="red.50" borderRadius="md" p={4}>
        <Text color="red.500">Error: {error}</Text>
      </Center>
    );
  }

  return (
    <Box position="relative" h="100%">
      {isLoading && (
        <Center position="absolute" top="0" left="0" right="0" bottom="0" bg="white" zIndex="1">
          <Spinner size="xl" />
        </Center>
      )}
      <Box
        ref={containerRef}
        width="100%"
        height="100%"
        border="1px solid"
        borderColor="gray.200"
        borderRadius="md"
        visibility={isLoading ? 'hidden' : 'visible'}
      />
      {properties && (
        <Box
          position="absolute"
          top="4"
          right="4"
          bg="white"
          p={2}
          borderRadius="md"
          boxShadow="md"
          fontSize="sm"
        >
          <Text fontWeight="bold">Properties:</Text>
          {Object.entries(properties).map(([key, value]) => (
            <Text key={key}>
              {key.replace(/_/g, ' ')}: {value}
            </Text>
          ))}
        </Box>
      )}
    </Box>
  );
};

export default MoleculeViewer;
